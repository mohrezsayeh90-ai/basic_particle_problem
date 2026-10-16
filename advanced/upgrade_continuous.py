"""Verify that only runtime/export orchestration changed before resuming old weights."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from build_notebook import source_payload


def recipe(source):
    tree=ast.parse(source);functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    result=[ast.dump(functions[n],include_attributes=False) for n in ['batch','objective']]
    main=functions['main']
    # Preserve model, RNG, optimizer and scheduler construction exactly.
    wanted={'rng','model','opt','scheduler'}
    for node in main.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in wanted for t in node.targets):
            result.append(ast.dump(node,include_attributes=False))
    loop=next(n for n in ast.walk(main) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step')
    for node in loop.body:
        if isinstance(node,ast.If) and any(isinstance(n,ast.Mod) for n in ast.walk(node.test)): break
        result.append(ast.dump(node,include_attributes=False))
    return result


def verify_upgrade(repo,previous_manifest,previous_commit):
    _,current,files=source_payload(repo)
    old=previous_manifest['files']
    for name,digest in old.items():
        if name.startswith(('advanced/','tests/')) or name=='run_parametric_pinn.py': continue
        if files.get(name)!=digest: raise RuntimeError('Physics dependency changed: '+name)
    old_source=subprocess.check_output(['git','show',previous_commit+':run_parametric_pinn.py'],cwd=repo,text=True)
    if recipe(old_source)!=recipe((repo/'run_parametric_pinn.py').read_text()):
        raise RuntimeError('Training algorithm changed; cannot treat this as an I/O-only upgrade.')
    return current
