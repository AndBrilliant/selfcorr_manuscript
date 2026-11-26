"""Executed in a subprocess per candidate: tolerant re-check of one frozen candidate.
argv: path to a JSON job {code, test, entry_point}. Prints one JSON line with per-assert outcomes."""
from pathlib import Path as _P
ROOT=_P(__file__).resolve().parents[1]
OUT=ROOT/"data/mbpp_v2/analysis"
import ast, json, sys, math
job=json.load(open(sys.argv[1]))
def norm(x, order):
    if isinstance(x,(list,tuple)):
        v=[norm(i,order) for i in x]
        return ('seq', tuple(sorted(v,key=repr)) if order else tuple(v))
    if isinstance(x,(set,frozenset)): return ('seq', tuple(sorted((norm(i,order) for i in x),key=repr)))
    if isinstance(x,dict): return ('dict', tuple(sorted(((repr(norm(k,order)),norm(v,order)) for k,v in x.items()))))
    if isinstance(x,bool): return ('num', float(x))
    if isinstance(x,(int,float)): return ('num', float(x))
    if isinstance(x,str): return ('str', x.strip())
    if hasattr(x,'__iter__') and not isinstance(x,(bytes,)):
        try: return norm(list(x),order)
        except Exception: pass
    return ('obj', repr(x))
def eq(a,b):
    if a[0]=='num' and b[0]=='num':
        return math.isclose(a[1],b[1],rel_tol=1e-6,abs_tol=1e-9) or (math.isnan(a[1]) and math.isnan(b[1]))
    if a[0]!=b[0]: return False
    if a[0] in ('seq','dict'):
        return len(a[1])==len(b[1]) and all(eq(x,y) if isinstance(x,tuple) and len(x)==2 and isinstance(x[0],str) and x[0] in('num','str','seq','dict','obj') else x==y for x,y in zip(a[1],b[1]))
    return a[1]==b[1]
g={}
try:
    exec(compile(job['code'],'<cand>','exec'),g)
    f=g[job['entry_point']]
except Exception as e:
    print(json.dumps({'load_error':type(e).__name__})); sys.exit()
tree=ast.parse(job['test'])
res=[]
for node in tree.body:
    if not isinstance(node,ast.Assert): 
        continue
    t=node.test; rec={'strict':False,'type':False,'order':False,'args':False,'parsed':False}
    def strict_eval():
        try: return bool(eval(compile(ast.Expression(t),'<t>','eval'),dict(g)))
        except Exception: return False
    rec['strict']=strict_eval()
    if isinstance(t,ast.Compare) and len(t.ops)==1 and isinstance(t.ops[0],ast.Eq):
        L,Rr=t.left,t.comparators[0]
        call=None
        if isinstance(L,ast.Call): call,exp=L,Rr
        elif isinstance(Rr,ast.Call): call,exp=Rr,L
        if call is not None:
            rec['parsed']=True
            try:
                expected=eval(compile(ast.Expression(exp),'<e>','eval'),dict(g))
                args=[eval(compile(ast.Expression(a),'<a>','eval'),dict(g)) for a in call.args]
                kw={k.arg:eval(compile(ast.Expression(k.value),'<k>','eval'),dict(g)) for k in call.keywords}
                fn=eval(compile(ast.Expression(call.func),'<f>','eval'),dict(g))
            except Exception:
                res.append(rec); continue
            outs=[]
            try: outs.append(('plain',fn(*args,**kw)))
            except Exception: pass
            if len(args)==1 and isinstance(args[0],(tuple,list)) and not kw:
                try: outs.append(('args',fn(*args[0])))
                except Exception: pass
            if len(args)>1 and not kw:
                for pack in (tuple,list):
                    try: outs.append(('args',fn(pack(args)))); break
                    except Exception: pass
            for how,o in outs:
                if how=='plain' and eq(norm(o,False),norm(expected,False)): rec['type']=True
                if how=='plain' and eq(norm(o,True),norm(expected,True)): rec['order']=True
                if how=='args' and (eq(norm(o,False),norm(expected,False)) or eq(norm(o,True),norm(expected,True))): rec['args']=True
    res.append(rec)
print(json.dumps({'asserts':res}))
