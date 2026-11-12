"""Label-free CPU retrieval and deterministic edge-only interventions."""
from collections import Counter
import hashlib
import json
import math
import random
import re


def tokens(text: str) -> list[str]:
    text=re.sub(r'([a-z0-9])([A-Z])',r'\1 \2',text)
    return re.findall(r'[a-z0-9]+',text.lower())


def bm25(query: str, candidates: dict, k1: float=1.2, b: float=0.75) -> dict[str,float]:
    if not candidates or k1 <= 0 or not 0 <= b <= 1:
        raise ValueError('BM25 requires nonempty candidates and valid parameters')
    documents={key:Counter(tokens(c['id']+' '+c['path']+' '+c['text'])) for key,c in candidates.items()}
    lengths={key:sum(counts.values()) for key,counts in documents.items()}
    average=sum(lengths.values())/len(lengths) or 1
    frequency=Counter(term for counts in documents.values() for term in counts)
    query_terms=set(tokens(query))
    return {key:sum(math.log(1+(len(documents)-frequency[t]+0.5)/(frequency[t]+0.5))*counts[t]*(k1+1)/(counts[t]+k1*(1-b+b*lengths[key]/average))
                    for t in sorted(query_terms) if counts[t]) for key,counts in documents.items()}


def edge_key(edge: dict) -> tuple:
    return edge['source'],edge['target'],edge.get('type',''),str(edge.get('key',0))


def seed_nodes(lexical: dict[str,float], seed_count: int) -> list[str]:
    """Single source of truth for the lexical seeds; one_hop and scoped corruption share it."""
    return sorted(lexical,key=lambda n:(-lexical[n],n))[:seed_count]


def delete_edges(edges: list[dict], fraction: float, seed: int) -> tuple[list[dict],dict]:
    if not 0 <= fraction <= 1:
        raise ValueError('Deletion fraction must be in [0,1]')
    ordered=sorted(edges,key=edge_key)
    if len({edge_key(e) for e in ordered}) != len(ordered):
        raise ValueError('Duplicate edge identities')
    permutation=list(range(len(ordered)))
    random.Random(seed).shuffle(permutation)
    count=math.floor(len(ordered)*fraction)
    removed=set(permutation[:count])
    kept=[dict(e) for i,e in enumerate(ordered) if i not in removed]
    return kept,{'original_edges':len(edges),'removed_edges':count,'remaining_edges':len(kept),
                 'actual_fraction':count/len(edges) if edges else 0,
                 'removed_ids_sha256':hashlib.sha256(json.dumps([edge_key(ordered[i]) for i in sorted(removed)]).encode()).hexdigest()}


def rewire_edges(edges: list[dict], fraction: float, seed: int) -> tuple[list[dict],dict]:
    """Rewire a seeded subset of edges to spurious targets; sources, types and counts are preserved."""
    if not 0 <= fraction <= 1:
        raise ValueError('Rewiring fraction must be in [0,1]')
    ordered=sorted(edges,key=edge_key)
    if len({edge_key(e) for e in ordered}) != len(ordered):
        raise ValueError('Duplicate edge identities')
    permutation=list(range(len(ordered)))
    random.Random(seed).shuffle(permutation)
    rewired_indices=sorted(permutation[:math.floor(len(ordered)*fraction)])
    pool=sorted({e['source'] for e in ordered}|{e['target'] for e in ordered})
    result=[dict(e) for e in ordered]
    taken={edge_key(e) for e in ordered}
    rewired=[]
    for index in rewired_indices:
        edge=ordered[index]
        # Per-edge RNG keeps a target stable regardless of how many other edges are rewired.
        candidates=[n for n in pool if n!=edge['source']]
        for target in random.Random(f'{seed}:{index}').sample(candidates,len(candidates)):
            candidate=dict(edge);candidate['target']=target
            if edge_key(candidate) not in taken:
                taken.discard(edge_key(edge));taken.add(edge_key(candidate))
                result[index]=candidate;rewired.append(edge_key(edge));break
    return result,{'original_edges':len(edges),'rewired_edges':len(rewired),
                   'actual_fraction':len(rewired)/len(edges) if edges else 0,
                   'rewired_ids_sha256':hashlib.sha256(json.dumps(sorted(rewired)).encode()).hexdigest()}


def _scoped(edges: list[dict], scope: set[str]) -> list[int]:
    """Indices of edges whose source is in the scope, in stable edge-key order."""
    ordered=sorted(edges,key=edge_key)
    return [i for i,e in enumerate(ordered) if e['source'] in scope]


def delete_scoped(edges: list[dict], fraction: float, seed: int, scope: set[str]) -> tuple[list[dict],dict]:
    """Delete only edges leaving the given seed scope; the fraction applies to that subset."""
    if not 0 <= fraction <= 1:
        raise ValueError('Deletion fraction must be in [0,1]')
    ordered=sorted(edges,key=edge_key)
    if len({edge_key(e) for e in ordered}) != len(ordered):
        raise ValueError('Duplicate edge identities')
    eligible=_scoped(ordered,scope)
    permutation=list(range(len(eligible)))
    random.Random(seed).shuffle(permutation)
    removed={eligible[i] for i in permutation[:math.floor(len(eligible)*fraction)]}
    kept=[dict(e) for i,e in enumerate(ordered) if i not in removed]
    return kept,{'original_edges':len(edges),'scope_edges':len(eligible),
                 'removed_edges':len(removed),'remaining_edges':len(kept),
                 'actual_fraction':len(removed)/len(eligible) if eligible else 0,
                 'removed_ids_sha256':hashlib.sha256(json.dumps([edge_key(ordered[i]) for i in sorted(removed)]).encode()).hexdigest()}


def rewire_scoped(edges: list[dict], fraction: float, seed: int, scope: set[str]) -> tuple[list[dict],dict]:
    """Rewire only edges leaving the given seed scope; sources, types and counts are preserved."""
    if not 0 <= fraction <= 1:
        raise ValueError('Rewiring fraction must be in [0,1]')
    ordered=sorted(edges,key=edge_key)
    if len({edge_key(e) for e in ordered}) != len(ordered):
        raise ValueError('Duplicate edge identities')
    eligible=_scoped(ordered,scope)
    permutation=list(range(len(eligible)))
    random.Random(seed).shuffle(permutation)
    chosen=sorted(eligible[i] for i in permutation[:math.floor(len(eligible)*fraction)])
    pool=sorted({e['source'] for e in ordered}|{e['target'] for e in ordered})
    result=[dict(e) for e in ordered]
    taken={edge_key(e) for e in ordered}
    rewired=[]
    for index in chosen:
        edge=ordered[index]
        candidates=[n for n in pool if n!=edge['source']]
        for target in random.Random(f'{seed}:{index}').sample(candidates,len(candidates)):
            candidate=dict(edge);candidate['target']=target
            if edge_key(candidate) not in taken:
                taken.discard(edge_key(edge));taken.add(edge_key(candidate))
                result[index]=candidate;rewired.append(edge_key(edge));break
    return result,{'original_edges':len(edges),'scope_edges':len(eligible),
                   'rewired_edges':len(rewired),'remaining_edges':len(result),
                   'actual_fraction':len(rewired)/len(eligible) if eligible else 0,
                   'rewired_ids_sha256':hashlib.sha256(json.dumps(sorted(rewired)).encode()).hexdigest()}


def one_hop(lexical: dict[str,float], edges: list[dict], seed_count: int, edge_budget: int, restart: float) -> tuple[dict,dict]:
    if seed_count <= 0 or edge_budget < 0 or not 0 <= restart <= 1:
        raise ValueError('Invalid expansion configuration')
    seeds=seed_nodes(lexical,seed_count)
    weights=[max(0,lexical[n]) for n in seeds]
    total=sum(weights)
    weights=[w/total for w in weights] if total else [1/len(seeds)]*len(seeds)
    adjacency={n:set() for n in seeds}
    for edge in edges:
        if edge['source'] in adjacency and edge['target'] in lexical:
            adjacency[edge['source']].add(edge['target'])
    # Only edges leaving the chosen seeds can influence the ranking; the rest of the
    # graph is invisible to this traversal however heavily it is corrupted.
    incident=sum(edge['source'] in adjacency for edge in edges)
    scores=dict.fromkeys(lexical,0.0)
    visited=0
    expanded=set()
    for seed,weight in zip(seeds,weights):
        neighbors=sorted(adjacency[seed])
        admitted=neighbors[:max(0,edge_budget-visited)]
        if not admitted:
            scores[seed]+=weight
            continue
        scores[seed]+=restart*weight
        # Unvisited mass remains on the seed; the visit cap never gives more weight to survivors.
        scores[seed]+=(1-restart)*weight*(1-len(admitted)/len(neighbors))
        for node in admitted:
            scores[node]+=(1-restart)*weight/len(neighbors)
            expanded.add(node)
        visited+=len(admitted)
    return scores,{'seed_nodes':seeds,'candidate_expansions':visited,'expanded_nodes':len(expanded-set(seeds)),
                   'edges_scanned':len(edges),'edge_budget':edge_budget,
                   'edges_incident_to_seeds':incident,
                   'distinct_seed_neighbors':sum(len(neighbors) for neighbors in adjacency.values())}


def rank_files(scores: dict, lexical: dict, candidates: dict) -> tuple[list[str],list[str]]:
    nodes=sorted(scores,key=lambda n:(-scores[n],-lexical[n],n))
    files=list(dict.fromkeys(candidates[n]['path'] for n in nodes))
    return files,nodes
