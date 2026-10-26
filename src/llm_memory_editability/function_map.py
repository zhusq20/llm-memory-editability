"""knowledge-reasoning-map-v1 (zero training): where in a released Qwen the computation is keyed by
the entity, by the question, or by both.

Prompts form a full factorial: entities e (single-token country names) x questions q (four
relations x two phrasings), all with the same token length, so every position lines up. For each
decoder layer l, token position p and quantity (residual stream after the layer, MLP address =
down_proj input, MLP output, mixer output), the vectors x[e, q] are split as

    x[e, q] = mu + a_e + b_q + c_eq,

a balanced two-way decomposition. Its sums of squares give three shares that add up to one:
entity (changes with the entity, same for every question), question (changes with the question,
same for every entity) and joint (depends on the pair, e.g. the answer). The question share is
further split into relation and phrasing parts.

The answer-supervised write of a counterfactual answer is decomposed per position as in
answer-gradient-address-v1: the rank-one term of position p in dL/dW_l has Frobenius norm
||delta_{l,p}|| ||k_{l,p}||.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from llm_memory_editability import read_interface_model as M

EOS = "<|im_end|>"
INSTRUCTION = "\nAnswer with only the name."
RELATIONS = ("capital", "continent", "language", "currency")
PHRASINGS = ("What is the {r} of {e}?", "Tell me the {r} of {e}.")
QUANTITIES = ("residual", "mlp_address", "mlp_output", "mixer_output")

AMERICAS = ["North America", "Central America", "Caribbean", "Americas"]
SOUTH = ["South America"]
OCEANIA = ["Oceania", "Australia"]
# country: (capital aliases, continent aliases, language aliases, currency aliases)
GOLD = {
    "France": (["Paris"], ["Europe"], ["French"], ["Euro"]),
    "Germany": (["Berlin"], ["Europe"], ["German"], ["Euro"]),
    "Italy": (["Rome"], ["Europe"], ["Italian"], ["Euro"]),
    "Spain": (["Madrid"], ["Europe"], ["Spanish"], ["Euro"]),
    "Portugal": (["Lisbon"], ["Europe"], ["Portuguese"], ["Euro"]),
    "Greece": (["Athens"], ["Europe"], ["Greek"], ["Euro"]),
    "Ireland": (["Dublin"], ["Europe"], ["Irish", "English"], ["Euro"]),
    "Netherlands": (["Amsterdam"], ["Europe"], ["Dutch"], ["Euro"]),
    "Belgium": (["Brussels"], ["Europe"], ["Dutch", "French", "German", "Flemish"], ["Euro"]),
    "Austria": (["Vienna"], ["Europe"], ["German"], ["Euro"]),
    "Finland": (["Helsinki"], ["Europe"], ["Finnish", "Swedish"], ["Euro"]),
    "Sweden": (["Stockholm"], ["Europe"], ["Swedish"], ["Krona", "kronor"]),
    "Norway": (["Oslo"], ["Europe"], ["Norwegian"], ["Krone", "kroner"]),
    "Denmark": (["Copenhagen"], ["Europe"], ["Danish"], ["Krone", "kroner"]),
    "Iceland": (["Reykjavik", "Reykjavík"], ["Europe"], ["Icelandic"], ["Krona", "Króna"]),
    "Poland": (["Warsaw"], ["Europe"], ["Polish"], ["Zloty", "Złoty"]),
    "Hungary": (["Budapest"], ["Europe"], ["Hungarian"], ["Forint"]),
    "Romania": (["Bucharest"], ["Europe"], ["Romanian"], ["Leu", "Lei"]),
    "Bulgaria": (["Sofia"], ["Europe"], ["Bulgarian"], ["Lev", "Euro"]),
    "Croatia": (["Zagreb"], ["Europe"], ["Croatian"], ["Euro", "Kuna"]),
    "Serbia": (["Belgrade"], ["Europe"], ["Serbian"], ["Dinar"]),
    "Ukraine": (["Kyiv", "Kiev"], ["Europe"], ["Ukrainian"], ["Hryvnia", "Hryvna"]),
    "Russia": (["Moscow"], ["Europe", "Asia"], ["Russian"], ["Ruble", "Rouble"]),
    "Switzerland": (
        ["Bern", "Berne"],
        ["Europe"],
        ["German", "French", "Italian", "Romansh"],
        ["Franc"],
    ),
    "Albania": (["Tirana"], ["Europe"], ["Albanian"], ["Lek"]),
    "Estonia": (["Tallinn"], ["Europe"], ["Estonian"], ["Euro"]),
    "Latvia": (["Riga"], ["Europe"], ["Latvian"], ["Euro"]),
    "Lithuania": (["Vilnius"], ["Europe"], ["Lithuanian"], ["Euro"]),
    "Slovakia": (["Bratislava"], ["Europe"], ["Slovak"], ["Euro"]),
    "Slovenia": (["Ljubljana"], ["Europe"], ["Slovene", "Slovenian"], ["Euro"]),
    "Belarus": (["Minsk"], ["Europe"], ["Belarusian", "Russian"], ["Ruble", "Rouble"]),
    "Japan": (["Tokyo"], ["Asia"], ["Japanese"], ["Yen"]),
    "China": (["Beijing"], ["Asia"], ["Chinese", "Mandarin"], ["Yuan", "Renminbi", "RMB"]),
    "India": (["New Delhi", "Delhi"], ["Asia"], ["Hindi", "English"], ["Rupee"]),
    "Pakistan": (["Islamabad"], ["Asia"], ["Urdu", "English"], ["Rupee"]),
    "Bangladesh": (["Dhaka"], ["Asia"], ["Bengali", "Bangla"], ["Taka"]),
    "Nepal": (["Kathmandu"], ["Asia"], ["Nepali"], ["Rupee"]),
    "Thailand": (["Bangkok"], ["Asia"], ["Thai"], ["Baht"]),
    "Vietnam": (["Hanoi"], ["Asia"], ["Vietnamese"], ["Dong", "Đồng"]),
    "Cambodia": (["Phnom Penh"], ["Asia"], ["Khmer"], ["Riel"]),
    "Laos": (["Vientiane"], ["Asia"], ["Lao"], ["Kip"]),
    "Myanmar": (["Naypyidaw", "Nay Pyi Taw", "Naypyitaw"], ["Asia"], ["Burmese"], ["Kyat"]),
    "Malaysia": (["Kuala Lumpur"], ["Asia"], ["Malay"], ["Ringgit"]),
    "Indonesia": (["Jakarta", "Nusantara"], ["Asia"], ["Indonesian"], ["Rupiah"]),
    "Philippines": (["Manila"], ["Asia"], ["Filipino", "Tagalog", "English"], ["Peso"]),
    "Singapore": (["Singapore"], ["Asia"], ["English", "Malay", "Mandarin", "Tamil"], ["Dollar"]),
    "Mongolia": (
        ["Ulaanbaatar", "Ulan Bator"],
        ["Asia"],
        ["Mongolian"],
        ["Tögrög", "Tugrik", "Togrog"],
    ),
    "Kazakhstan": (["Astana", "Nur-Sultan"], ["Asia"], ["Kazakh", "Russian"], ["Tenge"]),
    "Uzbekistan": (["Tashkent"], ["Asia"], ["Uzbek"], ["Som", "Sum"]),
    "Afghanistan": (["Kabul"], ["Asia"], ["Pashto", "Dari"], ["Afghani"]),
    "Iran": (["Tehran"], ["Asia"], ["Persian", "Farsi"], ["Rial"]),
    "Iraq": (["Baghdad"], ["Asia"], ["Arabic", "Kurdish"], ["Dinar"]),
    "Syria": (["Damascus"], ["Asia"], ["Arabic"], ["Pound", "Lira"]),
    "Lebanon": (["Beirut"], ["Asia"], ["Arabic"], ["Pound", "Lira"]),
    "Israel": (["Jerusalem"], ["Asia"], ["Hebrew"], ["Shekel"]),
    "Turkey": (["Ankara"], ["Asia", "Europe"], ["Turkish"], ["Lira"]),
    "Qatar": (["Doha"], ["Asia"], ["Arabic"], ["Riyal"]),
    "Kuwait": (["Kuwait City"], ["Asia"], ["Arabic"], ["Dinar"]),
    "Oman": (["Muscat"], ["Asia"], ["Arabic"], ["Rial"]),
    "Yemen": (["Sanaa", "Sana'a"], ["Asia"], ["Arabic"], ["Rial"]),
    "Armenia": (["Yerevan"], ["Asia", "Europe"], ["Armenian"], ["Dram"]),
    "Egypt": (["Cairo"], ["Africa"], ["Arabic"], ["Pound"]),
    "Morocco": (["Rabat"], ["Africa"], ["Arabic", "Berber", "Amazigh"], ["Dirham"]),
    "Algeria": (["Algiers"], ["Africa"], ["Arabic", "Berber"], ["Dinar"]),
    "Tunisia": (["Tunis"], ["Africa"], ["Arabic"], ["Dinar"]),
    "Libya": (["Tripoli"], ["Africa"], ["Arabic"], ["Dinar"]),
    "Sudan": (["Khartoum"], ["Africa"], ["Arabic", "English"], ["Pound"]),
    "Ethiopia": (["Addis Ababa"], ["Africa"], ["Amharic"], ["Birr"]),
    "Kenya": (["Nairobi"], ["Africa"], ["Swahili", "English"], ["Shilling"]),
    "Uganda": (["Kampala"], ["Africa"], ["English", "Swahili"], ["Shilling"]),
    "Tanzania": (["Dodoma", "Dar es Salaam"], ["Africa"], ["Swahili", "English"], ["Shilling"]),
    "Rwanda": (["Kigali"], ["Africa"], ["Kinyarwanda", "French", "English"], ["Franc"]),
    "Nigeria": (["Abuja"], ["Africa"], ["English"], ["Naira"]),
    "Ghana": (["Accra"], ["Africa"], ["English"], ["Cedi"]),
    "Senegal": (["Dakar"], ["Africa"], ["French", "Wolof"], ["Franc"]),
    "Angola": (["Luanda"], ["Africa"], ["Portuguese"], ["Kwanza"]),
    "Zambia": (["Lusaka"], ["Africa"], ["English"], ["Kwacha"]),
    "Zimbabwe": (["Harare"], ["Africa"], ["English", "Shona"], ["Dollar", "ZiG", "Gold"]),
    "Somalia": (["Mogadishu"], ["Africa"], ["Somali", "Arabic"], ["Shilling"]),
    "Brazil": (["Brasília", "Brasilia"], SOUTH, ["Portuguese"], ["Real"]),
    "Argentina": (["Buenos Aires"], SOUTH, ["Spanish"], ["Peso"]),
    "Chile": (["Santiago"], SOUTH, ["Spanish"], ["Peso"]),
    "Peru": (["Lima"], SOUTH, ["Spanish", "Quechua"], ["Sol"]),
    "Colombia": (["Bogotá", "Bogota"], SOUTH, ["Spanish"], ["Peso"]),
    "Venezuela": (["Caracas"], SOUTH, ["Spanish"], ["Bolívar", "Bolivar"]),
    "Bolivia": (["Sucre", "La Paz"], SOUTH, ["Spanish"], ["Boliviano"]),
    "Ecuador": (["Quito"], SOUTH, ["Spanish"], ["Dollar"]),
    "Uruguay": (["Montevideo"], SOUTH, ["Spanish"], ["Peso"]),
    "Paraguay": (["Asunción", "Asuncion"], SOUTH, ["Spanish", "Guarani"], ["Guaraní", "Guarani"]),
    "Mexico": (["Mexico City"], AMERICAS, ["Spanish"], ["Peso"]),
    "Canada": (["Ottawa"], AMERICAS, ["English", "French"], ["Dollar"]),
    "Cuba": (["Havana"], AMERICAS, ["Spanish"], ["Peso"]),
    "Jamaica": (["Kingston"], AMERICAS, ["English"], ["Dollar"]),
    "Haiti": (["Port-au-Prince"], AMERICAS, ["French", "Creole"], ["Gourde"]),
    "Panama": (["Panama City"], AMERICAS, ["Spanish"], ["Balboa", "Dollar"]),
    "Honduras": (["Tegucigalpa"], AMERICAS, ["Spanish"], ["Lempira"]),
    "Guatemala": (["Guatemala City"], AMERICAS, ["Spanish"], ["Quetzal"]),
    "Nicaragua": (["Managua"], AMERICAS, ["Spanish"], ["Córdoba", "Cordoba"]),
    "Australia": (["Canberra"], OCEANIA, ["English"], ["Dollar"]),
    "Fiji": (["Suva"], OCEANIA, ["English", "Fijian"], ["Dollar"]),
}


# ---------------------------------------------------------------- prompts


def question_texts():
    """The eight questions in a fixed order: relation-major, phrasing-minor."""
    return [(r, t) for r in RELATIONS for t in range(len(PHRASINGS))]


def prompt_text(entity, relation, phrasing):
    return PHRASINGS[phrasing].format(r=relation, e=entity) + INSTRUCTION


def single_token_entities(tok, candidates=None):
    """Entities that are one token after a space, the way they appear in every prompt."""
    out = []
    for e in candidates or GOLD:
        if len(tok(" " + e, add_special_tokens=False)["input_ids"]) == 1:
            out.append(e)
    return out


def encode_grid(tok, entities):
    """Token ids for every (entity, question) chat prompt; all rows must share one length.

    Returns ids (E, Q, T), the index of the first question token, the entity and relation token
    positions, and readable labels for the analysed span (first question token .. last token)."""
    questions = question_texts()
    rows, starts, e_pos, r_pos = [], set(), set(), set()
    for e in entities:
        row = []
        for r, t in questions:
            user = prompt_text(e, r, t)
            text = M.chat(tok, user, False)
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            u0 = text.index(user)
            q = PHRASINGS[t].format(r=r, e=e)
            ec = u0 + q.index(" " + e) + 1
            rc = u0 + q.index(" " + r) + 1
            starts.add(next(i for i, (a, b) in enumerate(offs) if b > u0))
            e_pos.update(i for i, (a, b) in enumerate(offs) if a < ec + len(e) and b > ec)
            r_pos.update(i for i, (a, b) in enumerate(offs) if a < rc + len(r) and b > rc)
            row.append(ids)
        rows.append(row)
    lengths = {len(x) for row in rows for x in row}
    if len(lengths) != 1 or len(starts) != 1 or len(e_pos) != 1 or len(r_pos) != 1:
        raise ValueError(f"prompts do not align: {lengths} {starts} {e_pos} {r_pos}")
    ids = torch.tensor(rows)
    start = starts.pop()
    labels = [tok.decode([i]) for i in rows[0][0][start:]]
    return ids, start, e_pos.pop(), r_pos.pop(), labels


def counterfactual_targets(entities):
    """For each (entity, relation): the first alias of the next entity (cyclically) whose
    first alias differs from every alias of this entity. Deterministic, model independent."""
    out = {}
    pool = list(entities) + [g for g in GOLD if g not in entities]
    for i, e in enumerate(entities):
        for j, r in enumerate(RELATIONS):
            own = {a.lower() for a in GOLD[e][j]}
            for k in range(1, len(pool)):
                cand = GOLD[pool[(i + k) % len(pool)]][j][0]
                if cand.lower() not in own:
                    out[(e, r)] = cand
                    break
    return out


def correct(text, entity, relation):
    t = text.lower()
    return any(a.lower() in t for a in GOLD[entity][RELATIONS.index(relation)])


# ---------------------------------------------------------------- decomposition


def two_way_shares(x):
    """x (E, Q, P, d) -> dict of (P,) shares: entity, question, joint (sum to one), and the
    question share split into relation and phrasing parts when Q factors as relations x
    phrasings (relation-major). Uses float64; positions with zero variance give NaN."""
    x = x.double()
    E, Q = x.shape[:2]
    mu = x.mean((0, 1))
    a = x.mean(1) - mu  # (E, P, d)
    b = x.mean(0) - mu  # (Q, P, d)
    c = x - mu - a[:, None] - b[None]
    ss_e = Q * a.pow(2).sum((0, -1))
    ss_q = E * b.pow(2).sum((0, -1))
    ss_j = c.pow(2).sum((0, 1, -1))
    total = ss_e + ss_q + ss_j
    nan = torch.full_like(total, float("nan"))
    ok = total > 1e-12 * (x.pow(2).sum((0, 1, -1)) + 1e-30)

    def share(s):
        return torch.where(ok, s / total.clamp_min(1e-300), nan)

    out = dict(entity=share(ss_e), question=share(ss_q), joint=share(ss_j))
    n_t = len(PHRASINGS)
    if Q == len(RELATIONS) * n_t:
        b_r = b.view(len(RELATIONS), n_t, *b.shape[1:]).mean(1)  # relation main effect
        ss_r = E * n_t * b_r.pow(2).sum((0, -1))
        out["relation"] = share(ss_r)
        out["phrasing"] = share(ss_q - ss_r)
    out["total_ss"] = total
    return out


def similarity_matrix(x):
    """x (n, d) -> (n, n) cosine similarities of the centred vectors (float64)."""
    x = x.double()
    x = x - x.mean(0)
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    return x @ x.T


# ---------------------------------------------------------------- capture


class Capture:
    """Forward hooks on every decoder layer: MLP address (down_proj input), MLP output
    (down_proj output) and mixer output (self_attn or linear_attn), restricted to positions
    [start:]. The residual stream comes from output_hidden_states."""

    def __init__(self, model, start):
        self.model, self.start = model, start
        self.store, self.handles = {}, []

    def __enter__(self):
        for i, layer in enumerate(M.layers(self.model)):

            def down(module, args, output, i=i):
                self.store[("mlp_address", i)] = args[0][:, self.start :].detach().float()
                self.store[("mlp_output", i)] = output[:, self.start :].detach().float()

            def mixer(module, args, output, i=i):
                y = output[0] if isinstance(output, tuple) else output
                self.store[("mixer_output", i)] = y[:, self.start :].detach().float()

            self.handles.append(layer.mlp.down_proj.register_forward_hook(down))
            mix = layer.linear_attn if hasattr(layer, "linear_attn") else layer.self_attn
            self.handles.append(mix.register_forward_hook(mixer))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()


@torch.no_grad()
def forward_states(model, ids, start, batch=32):
    """ids (N, T) unpadded rows of one length -> {quantity: (L, N, P, d)} on CPU float32, plus
    greedy next-token ids at the last position."""
    device = next(model.parameters()).device
    n_layers = len(M.layers(model))
    out = {q: [[] for _ in range(n_layers)] for q in QUANTITIES}
    greedy = []
    for i in range(0, len(ids), batch):
        x = ids[i : i + batch].to(device)
        with Capture(model, start) as cap:
            res = model.model(input_ids=x, use_cache=False, output_hidden_states=True)
        hs = res.hidden_states
        for layer in range(n_layers):
            out["residual"][layer].append(hs[layer + 1][:, start:].float().cpu())
            for q in QUANTITIES[1:]:
                out[q][layer].append(cap.store[(q, layer)].cpu())
        greedy.append(model.lm_head(res.last_hidden_state[:, -1]).argmax(-1).cpu())
    return {q: torch.stack([torch.cat(v) for v in out[q]]) for q in QUANTITIES}, torch.cat(greedy)


@torch.no_grad()
def generate(model, tok, ids, max_new_tokens=12, batch=64):
    """Greedy continuations of equal-length rows (no padding needed)."""
    device = next(model.parameters()).device
    texts = []
    for i in range(0, len(ids), batch):
        x = ids[i : i + batch].to(device)
        y = model.generate(
            input_ids=x,
            attention_mask=torch.ones_like(x),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        texts += tok.batch_decode(y[:, x.shape[1] :], skip_special_tokens=True)
    return texts


def write_shares(model, tok, ids, targets, start, batch=16):
    """Per-row answer-supervised write of a counterfactual reply (target + <|im_end|>), summed
    reply-token cross entropy as in M2. Returns approx shares (L, N, P + 1): the per-position
    ||delta|| ||k|| over the analysed prompt span (last prompt token included), with the reply
    positions that predict later reply tokens pooled into the last column, normalised over all
    positions of the row."""
    from llm_memory_editability.answer_gradient import DownProjCapture

    device = next(model.parameters()).device
    n_layers = len(M.layers(model))
    shares = []
    for i in range(0, len(ids), batch):
        prompt = ids[i : i + batch]
        reply = [
            tok(t + EOS, add_special_tokens=False)["input_ids"] for t in targets[i : i + batch]
        ]
        width = prompt.shape[1] + max(map(len, reply))
        full = torch.full((len(prompt), width), tok.pad_token_id or 0, dtype=torch.long)
        mask = torch.zeros_like(full)
        for j, r in enumerate(reply):
            row = prompt[j].tolist() + r
            full[j, : len(row)] = torch.tensor(row)  # right padding; causal, so prompt is exact
            mask[j, : len(row)] = 1
        full, mask = full.to(device), mask.to(device)
        T = prompt.shape[1]
        with torch.enable_grad(), DownProjCapture(model) as cap:
            hidden = model.model(input_ids=full, attention_mask=mask, use_cache=False)
            hidden = hidden.last_hidden_state
            loss = 0.0
            for j, r in enumerate(reply):
                logits = model.lm_head(hidden[j, T - 1 : T - 1 + len(r)]).float()
                loss = loss + F.cross_entropy(
                    logits, torch.tensor(r, device=device), reduction="sum"
                )
            loss.backward()
            k = torch.stack([cap.k[i] for i in range(n_layers)])  # (L, B, W, d_ff)
            d = torch.stack([cap.delta[i] for i in range(n_layers)])
        norm = (k.norm(dim=-1) * d.norm(dim=-1)).double()  # (L, B, W)
        for j, r in enumerate(reply):
            nj = norm[:, j, : T - 1 + len(r)]  # every position up to the last predicting one
            total = nj.sum(-1, keepdim=True)
            span = nj[:, start:T]  # analysed prompt span, last prompt token included
            later = nj[:, T : T - 1 + len(r)].sum(-1, keepdim=True)  # reply predicts reply
            shares.append(torch.cat([span, later], -1) / total)
        del k, d, hidden
    return torch.stack(shares, 1).cpu()  # (L, N, P_span + 1); last column = later reply tokens
