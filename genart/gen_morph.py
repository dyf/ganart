import random
import numpy as np
import h5py

from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from allensdk.core.cell_types_cache import CellTypesCache

def resample_branch(branch, segment_length=1.0):
    p = np.array([ [ c['x'], c['y'], c['z'] ] for c in branch ])

    branch_len, t = branch_length(p)

    N = int(np.round(branch_len / segment_length))

    spline = CubicSpline(t/branch_len, p)
    
    even_t = np.linspace(0, 1, N)
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
                
            yield branch[::-1], c['type']

    
def recon_to_commands(recon, max_cmds, segment_length=2.0):
    commands = np.zeros([max_cmds+1, 8])
    p_prev = None
    cmd_i = 0

    ctype_index = {
        3: 7,
        4: 7
    }
    

    for branch, ctype in branch_iter(recon):
        # skip axons
        if ctype == 2:
            continue
        
        p = resample_branch(branch, segment_length=segment_length)

        # add point from previous branch
        if p_prev is None:
            p_prev = p[0]

        p = np.vstack( [ p_prev, p ] )

        diff = np.diff(p, axis=0)

        commands[cmd_i:cmd_i+diff.shape[0], :3] = diff
        commands[cmd_i,3:6] = [ 1, 0, 0 ]
        commands[cmd_i+1:cmd_i+diff.shape[0], 3:6] = [ 0, 1, 0 ]

        cidx = ctype_index[ctype]
        commands[cmd_i+1:cmd_i+diff.shape[0], cidx] = 1

        p_prev = p[-1]
        cmd_i += diff.shape[0]

    commands[cmd_i:] = [ 0, 0, 0, 0, 0, 1, 0, 0 ]

    # first command is a no-op
    return commands[1:]
    
    
def gen_metadata(cells):
    keys = [ 'species', 'structure_layer_name', 'structure_area_abbrev', 'dendrite_type', 'structure_hemisphere' ]
    values = [ sorted(list(set([ c[key] for c in cells ]))) for key in keys ]
    lens = [ len(v) for v in values ]
    starts = np.cumsum([0] + lens[:-1])

    for c in cells:
        vec = np.zeros(np.sum(lens))

        for ki, k in enumerate(keys):
            v = c[k]
            idx = values[ki].index(v)
            vec[starts[ki]+idx] = 1

            print(ki, k, v, values[ki], idx, starts[ki])

        print(vec)
        break

def transform_recon(recon, rotate=True, noise_scale=0.1):
    
    # move soma to origin
    o = recon.soma
    tm = np.array([[1,0,0,-o['x']],
                   [0,1,0,-o['y']],
                   [0,0,1,-o['z']],
                   [0,0,0,1]])

    # rotate randomly
    rm = np.eye(4)
    
    if rotate:
        angle = np.random.random() * 2 * np.pi
        axis = np.random.random(3)
        axis /= np.linalg.norm(axis)
    
        r = Rotation.from_rotvec(angle * axis)

        rm[:3,:3] = r.as_matrix()

    xfm = np.dot(rm, tm)

    for c in recon.compartment_list:
        p = np.array([c['x'], c['y'], c['z'], 1])
        pn = np.dot(xfm, p)

        if noise_scale:
            pn[:3] += np.random.random(3)*noise_scale - noise_scale*0.5

        c['x'] = pn[0]
        c['y'] = pn[1]
        c['z'] = pn[2]
        
    
def main():
    n_copies = 10
    
    ctc = CellTypesCache(manifest_file='./ctc/manifest.json')

    cells = ctc.get_cells(require_reconstruction=True)

    #gen_metadata(cells)

    max_cmds = 10000
    all_cmds = np.zeros((len(cells)*n_copies, max_cmds, 8))

    i = 0
    for ci,c in enumerate(cells):
        for _ in range(n_copies):
            print(ci, i)
            r = ctc.get_reconstruction(c['id'])
            transform_recon(r, rotate=True, noise_scale=0.1)

            all_cmds[i] = recon_to_commands(r, max_cmds, segment_length=2.5)
            i += 1

    with h5py.File("morphologies.h5","w") as f:
        ds = f.create_dataset("data", data=all_cmds)
        

if __name__ == "__main__": main()
