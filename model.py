"""
Small GPT-style decoder-only Transformer, conditioned on a fixed-size
molecule-property vector (class one-hot + normalized MW/LogP/HBA/HBD/PSA/RotB).

Conditioning is injected two ways (both cheap and effective at this data scale):
  1. Projected into a "condition token" prepended to the sequence (FiLM-lite: the
     model can attend back to it at every layer via causal self-attention).
  2. Added to every token embedding (a constant bias per-sequence), which keeps
     the conditioning signal from being "forgotten" over long generations.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(attn_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out))


class Block(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class ConditionalSMILESTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        cond_dim,
        max_len,
        d_model=256,
        n_layer=6,
        n_head=8,
        d_ff=1024,
        dropout=0.1,
        pad_idx=0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.max_len = max_len
        self.seq_len_with_cond = max_len + 1  # +1 for prepended condition token

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len_with_cond, d_model))
        self.cond_proj_token = nn.Linear(cond_dim, d_model)   # -> prepended token
        self.cond_proj_bias = nn.Linear(cond_dim, d_model)    # -> added to every token
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, d_ff, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(self.seq_len_with_cond, self.seq_len_with_cond)).view(
                1, 1, self.seq_len_with_cond, self.seq_len_with_cond
            ),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, token_ids, cond, targets=None):
        """
        token_ids: (B, T) int64, T == max_len
        cond:      (B, cond_dim) float32
        targets:   (B, T) int64 or None
        """
        B, T = token_ids.shape
        cond_token = self.cond_proj_token(cond).unsqueeze(1)     # (B, 1, D)
        cond_bias = self.cond_proj_bias(cond).unsqueeze(1)       # (B, 1, D)

        tok = self.tok_emb(token_ids) + cond_bias                # (B, T, D)
        x = torch.cat([cond_token, tok], dim=1)                  # (B, T+1, D)
        x = x + self.pos_emb[:, : x.size(1), :]
        x = self.drop(x)

        mask = self.causal_mask[:, :, : x.size(1), : x.size(1)]
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        logits = self.head(x)          # (B, T+1, V)
        # Position 0 is the condition token (no input token yet); positions
        # 1..T are the outputs after seeing input tokens 0..T-1 respectively.
        # logits[:, i, :] (post-slice, i.e. full-sequence position i+1) is the
        # prediction after consuming input token i, which should predict
        # target[i] = the *next* character. So we drop position 0, not -1.
        logits = logits[:, 1:, :]      # (B, T, V), aligned with targets

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.pad_idx,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, cond, stoi, itos, max_new_tokens, temperature=1.0,
                 top_k=None, top_p=None, device="cpu"):
        """
        Autoregressively sample one sequence per row of `cond`.

        top_k: restrict sampling to the k most likely next tokens.
        top_p: nucleus sampling — restrict to the smallest set of tokens whose
               cumulative probability exceeds p (e.g. 0.9). Adapts the
               candidate pool size per-step (unlike fixed top_k), which tends
               to avoid both "boring" high-confidence loops and wild
               low-probability derailments. top_k and top_p can be combined
               (top_k applied first, then top_p within the reduced set).
        """
        self.eval()
        B = cond.size(0)
        pad_idx = stoi["<pad>"]
        bos_idx = stoi["<bos>"]
        eos_idx = stoi["<eos>"]

        tokens = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens - 1):
            T = tokens.size(1)
            padded = torch.full((B, self.max_len), pad_idx, dtype=torch.long, device=device)
            padded[:, :T] = tokens
            logits, _ = self.forward(padded, cond)
            next_logits = logits[:, T - 1, :] / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                # keep smallest prefix with cumulative prob > top_p; always keep first token
                sorted_remove = cum_probs > top_p
                sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
                sorted_remove[:, 0] = False
                remove_mask = torch.zeros_like(sorted_remove).scatter(1, sorted_idx, sorted_remove)
                next_logits = next_logits.masked_fill(remove_mask, float("-inf"))

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token[finished] = pad_idx
            tokens = torch.cat([tokens, next_token], dim=1)

            finished = finished | (next_token.squeeze(1) == eos_idx)
            if finished.all():
                break

        return self._tokens_to_strings(tokens, itos, eos_idx, pad_idx)

    @torch.no_grad()
    def beam_search(self, cond_single, stoi, itos, max_new_tokens, beam_width=10, device="cpu"):
        """
        Deterministic beam search for ONE conditioning vector (cond_single:
        shape (cond_dim,)). Returns the beam_width candidate strings ranked by
        total log-probability, most likely first.

        Useful when you want the model's single "best guess" for a given
        class/property target rather than a diverse sample (e.g. for a final
        candidate shortlist) — complements generate()'s stochastic sampling,
        which is better for exploring diverse chemical space.
        """
        self.eval()
        pad_idx = stoi["<pad>"]
        bos_idx = stoi["<bos>"]
        eos_idx = stoi["<eos>"]

        cond = cond_single.unsqueeze(0).repeat(beam_width, 1).to(device)
        beams = torch.full((beam_width, 1), bos_idx, dtype=torch.long, device=device)
        beam_scores = torch.zeros(beam_width, device=device)
        beam_scores[1:] = float("-inf")  # only the first beam is "real" at step 0
        finished = torch.zeros(beam_width, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens - 1):
            T = beams.size(1)
            padded = torch.full((beam_width, self.max_len), pad_idx, dtype=torch.long, device=device)
            padded[:, :T] = beams
            logits, _ = self.forward(padded, cond)
            log_probs = F.log_softmax(logits[:, T - 1, :], dim=-1)  # (beam_width, V)

            # finished beams can only extend with pad (score unchanged)
            log_probs[finished] = float("-inf")
            log_probs[finished, pad_idx] = 0.0

            candidate_scores = beam_scores.unsqueeze(1) + log_probs  # (beam_width, V)
            flat = candidate_scores.view(-1)
            top_scores, top_idx = torch.topk(flat, beam_width)
            vocab_size = log_probs.size(-1)
            src_beam = top_idx // vocab_size
            next_tok = top_idx % vocab_size

            beams = torch.cat([beams[src_beam], next_tok.unsqueeze(1)], dim=1)
            beam_scores = top_scores
            finished = finished[src_beam] | (next_tok == eos_idx)
            if finished.all():
                break

        order = torch.argsort(beam_scores, descending=True)
        beams = beams[order]
        strings = self._tokens_to_strings(beams, itos, eos_idx, pad_idx)
        return list(zip(strings, beam_scores[order].tolist()))

    @staticmethod
    def _tokens_to_strings(tokens, itos, eos_idx, pad_idx):
        results = []
        for row in tokens.tolist():
            chars = []
            for t in row[1:]:  # skip BOS
                if t == eos_idx or t == pad_idx:
                    break
                chars.append(itos[t])
            results.append("".join(chars))
        return results
