import random
import numpy as np
from scipy.interpolate import CubicSpline

from allensdk.core.cell_types_cache import CellTypesCache

def resample_branch(branch, factor=1):
    p = np.array([ [ c['x'], c['y'], c['z'] ] for c in branch ])
    branch_len, t = branch_length(p)

    spline = CubicSpline(t/branch_len, p)
    
    even_t = np.linspace(0, 1, int(len(branch)*factor))
    even_p = spline(even_t)
    
    return even_p
                    
def branch_length(p):
    diff = np.diff(p, axis=0)
    dist = np.linalg.norm(diff, axis=1)
    return dist.sum(), np.cumsum(np.hstack([[0],dist]))

def recon_iter(recon, shuffle_children=True):
    to_visit = [recon.soma]

    while to_visit:
        c = to_visit.pop(0)

        children = [ recon.node(cid) for cid in c['children'] ]

        if shuffle_children:
            random.shuffle(children)

        to_visit += children

        yield c
    
def branch_iter(recon, shuffle_children=True):
    for c in recon_iter(recon, shuffle_children):

        # when we hit a branch point, backtrace to previous branch point
        if len(c['children']) != 1 and c != recon.soma:
            branch = [c]
            parent = recon.node(c['parent'])

            while True:
                branch.append(parent)
                if len(parent['children']) > 1:
                    break
                parent = recon.node(parent['parent'])
                
            yield branch[::-1]

    
def recon_to_commands(recon, max_cmds, resample_factor=2):
    commands = np.zeros([max_cmds, 6])
    p_prev = None
    cmd_i = 0

    total = 0
    for branch in branch_iter(recon):
        p = resample_branch(branch, factor=resample_factor)
        total += p.shape[0]

        # add point from previous branch
        if p_prev is None:
            p_prev = p[0]

        p = np.vstack( [ p_prev, p ] )

        diff = np.diff(p, axis=0)

        commands[cmd_i:cmd_i+diff.shape[0], :3] = diff
        commands[cmd_i,3:] = [ 1, 0, 0 ]
        commands[cmd_i+1:cmd_i+diff.shape[0], 3:] = [ 0, 1, 0 ]

        p_prev = p[-1]
        cmd_i += diff.shape[0]

    commands[cmd_i:] = [ 0, 0, 0, 0, 0, 1 ]

    return commands
    
    
def commands_to_recon(commands):
    pass

def main():
    ctc = CellTypesCache(manifest_file='./ctc/manifest.json')

    cells = ctc.get_cells(require_reconstruction=True)
    
    for c in cells:
        r = ctc.get_reconstruction(c['id'])
        print(len(r.compartment_list))
        cmds = recon_to_commands(r, 4000)
        print(cmds)
        break
        

if __name__ == "__main__": main()
