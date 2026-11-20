"""Fixed causal RGB support. Reference start and target duration never move."""
def shot_intervals(row):
    end=row['shot_spans'][-1][1]
    return [(4*a,4*b-(3 if b==end else 0)) for a,b in row['shot_spans']]
def reference_interval(ref,timeline_end):
    a,b=map(int,ref['dst_range'])
    if not 0<=a<b<=timeline_end: raise ValueError('Invalid reference destination')
    return 4*a,4*b-(3 if b==timeline_end else 0)
