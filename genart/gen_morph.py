import random
import numpy as np
import h5py

from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from allensdk.core.cell_types_cache import CellTypesCache

COMMAND_TYPES = ["move", "draw", "end"]
COMMAND_TYPE_TO_ONE_HOT = {
    "move": (1, 0, 0),
    "draw": (0, 1, 0),
    "end": (0, 0, 1)
}

SEG_TYPES = [3,4,None]
SEG_TYPE_TO_ONE_HOT = {
    3: (1, 0, 0),
    4: (0, 1, 0),
    None: (0, 0, 1)
}

    
def toy_morph():
    from allensdk.core.swc import Morphology, Compartment

    compartments = [ Compartment(id=0, x=1, y=1, z=1, radius=1, type=1, parent=-1),
                     Compartment(id=1, x=2, y=1, z=1, radius=1, type=3, parent=0),
                     Compartment(id=2, x=3, y=1, z=1, radius=1, type=3, parent=1),
                     Compartment(id=3, x=4, y=1, z=1, radius=1, type=3, parent=2),
                     Compartment(id=4, x=4, y=2, z=1, radius=1, type=4, parent=3),
                     Compartment(id=5, x=4, y=0, z=1, radius=1, type=4, parent=3),
                     Compartment(id=6, x=1, y=1, z=2, radius=1, type=3, parent=0) ]

    return Morphology(compartment_list=compartments)

def resample_branch(branch, segment_length=1.0):
    p = np.array([ [ c['x'], c['y'], c['z'] ] for c in branch ])
    
    branch_len, t = branch_length(p)    

    N = max(2, int(np.round(branch_len / segment_length) + 1))

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

            while parent:
                branch.append(parent)
                if len(parent['children']) > 1:
                    break
                parent = recon.node(parent['parent'])
                
            yield branch[::-1], c['type']

def commands_to_recon(cmds):
    from allensdk.core.swc import Morphology, Compartment
    
    compartments = [ Compartment(id=0, x=0, y=0, z=0, type=1, radius=1, parent=-1) ]

    p_prev = np.array([0,0,0])
    id_prev = 0
    
    for cmd in cmds:
        cmd_type = COMMAND_TYPES[np.argmax(cmd[3:6])]
        if cmd_type == "end":
            break
        
        seg_type = SEG_TYPES[np.argmax(cmd[6:9])]
        p_delta = cmd[:3]

        p = p_prev + p_delta
        cid = id_prev + 1
        c = Compartment(id=cid, x=p[0], y=p[1], z=p[2], type=seg_type, parent=-1, radius=1)

        if cmd_type == "draw":
            c['parent'] = id_prev

        compartments.append(c)
        p_prev = p

    return Morphology(compartment_list=compartments)

        
            


    return Morphology(compartment_list=[Compartment(c) for c in compartments])

def recon_to_commands(recon, max_cmds, segment_length=2.0):
    commands = np.zeros([max_cmds+1, 9])

    commands[:,:] = [ 0, 0, 0, 0, 0, 1, 0, 0, 1 ]
    
    p_prev = None
    cmd_i = 0    

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

        ctype_oh = SEG_TYPE_TO_ONE_HOT[ctype]
        commands[cmd_i:cmd_i+diff.shape[0], 6:9] = ctype_oh

        p_prev = p[-1]
        cmd_i += diff.shape[0]



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

def transform_recon(recon, rotate=True, scale=0.001, noise_scale=0.1):
    p = np.array([ [c['x'], c['y'], c['z']] for c in recon.compartment_list])
    size = np.max(p,axis=0) - np.min(p,axis=0)    
    
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

    # shrink
    sc = np.array([[ scale, 0, 0, 0],
                   [ 0, scale, 0, 0],
                   [ 0, 0, scale, 0],
                   [ 0, 0, 0,     1]])

    xfm = np.dot(sc, np.dot(rm, tm))

    for c in recon.compartment_list:
        
        
        p = np.array([c['x'], c['y'], c['z'], 1])
        pn = np.dot(xfm, p)

        if noise_scale and c is not recon.soma:
            pn[:3] += np.random.random(3)*noise_scale - noise_scale*0.5

        c['x'] = pn[0]
        c['y'] = pn[1]
        c['z'] = pn[2]
        
    
def main():
    n_copies = 10
    
    ctc = CellTypesCache(manifest_file='./ctc/manifest.json')

    cells = ctc.get_cells(require_reconstruction=True)

    #gen_metadata(cells)

    max_cmds = 1000
    all_cmds = np.zeros((len(cells)*n_copies, max_cmds, 8))

    i = 0
    for ci,c in enumerate(cells):
        for _ in range(n_copies):
            print(ci, i)
            r = ctc.get_reconstruction(c['id'])
            transform_recon(r, rotate=True, scale=0.001, noise_scale=0.00001)

            all_cmds[i] = recon_to_commands(r, max_cmds, segment_length=0.025)
            i += 1

    with h5py.File("morphologies.h5","w") as f:
        ds = f.create_dataset("data", data=all_cmds)
        

if __name__ == "__main__": main()
