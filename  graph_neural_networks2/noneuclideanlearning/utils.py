import os
import numpy as np
from random import choice
from IPython.display import HTML
import trimesh

## Note: Non-smooth surfaces or bad triangulations may lead to non-spiral orderings of the vertices.
## Common issue in badly triangulated surfaces is that there exist some edges that belong to more than two triangles. In this
## case the mathematical definition of the spiral is insufficient. In this case, in this version of the code, we randomly
## choose two triangles in order to continue the inductive assignment of the order to the rest of the vertices.

def show_mesh(vertex_positions, width=500, height=500):
    new_mesh = trimesh.load('DFAUST/template/template.obj', process = False)
    new_mesh.vertices = vertex_positions
    mesh_html = new_mesh.show(height=height).data
    # <--- trimesh developpers had an alternative hand attachment points
    #  specify plot width
    mesh_html = mesh_html.replace('width="100%"', f'width="{width}px"', 1)
    #  fix broken aspect ratio
    mesh_html = mesh_html.replace('render();window.addEventListener', 'camera.aspect=window.innerWidth/window.innerHeight;camera.updateProjectionMatrix();renderer.setSize(window.innerWidth,window.innerHeight);controls.handleResize();render();window.addEventListener')
    #  auto fit viewport to object
    mesh_html = mesh_html.replace('render();window.addEventListener', 'autoFit(scene,camera,controls);render();window.addEventListener')
    #  make zooming more responsive
    mesh_html = mesh_html.replace('zoomSpeed=1.2', 'zoomSpeed=5')
    
    return HTML(mesh_html)


def get_settings(root_dir, dataset, name):
    args = {}

    # below are the arguments for the DFAUST run
    reference_mesh_file = os.path.join(root_dir, dataset, 'template', 'template.obj')
    downsample_directory = os.path.join(root_dir, dataset,'template', 'COMA_downsample')

    ds_factors = [4, 4, 4, 4]
    step_sizes = [2, 2, 1, 1, 1]
    filter_sizes_enc = [[3, 16, 32, 64, 128],[[],[],[],[],[]]]
    filter_sizes_dec = [[128, 64, 32, 32, 16],[[],[],[],[],3]]

    reference_points = [[414]]

    args = {'generative_model': 'autoencoder',
            'name': name, 'data': os.path.join(root_dir, dataset, 'preprocessed',name),
            'results_folder':  os.path.join(root_dir, dataset,'results/spirals_autoencoder'),
            'reference_mesh_file':reference_mesh_file, 'downsample_directory': downsample_directory,
            'checkpoint_file': 'checkpoint',
            'seed':2, 'loss':'l1',
            'batch_size':16, 'num_epochs':300, 'eval_frequency':200, 'num_workers': 4,
            'filter_sizes_enc': filter_sizes_enc, 'filter_sizes_dec': filter_sizes_dec,
            'nz':16, 
            'ds_factors': ds_factors, 'step_sizes' : step_sizes, 'dilation': [2, 2, 1, 1, 1],

            'lr':1e-3, 
            'regularization': 5e-5,         
            'scheduler': True, 'decay_rate': 0.99,'decay_steps':1,  
            'resume': False,

            'mode':'train', 'shuffle': True, 'nVal': 500, 'normalization': True}
    args['results_folder'] = os.path.join(args['results_folder'],'latent_'+str(args['nz']))
    
    if not os.path.exists(os.path.join(args['results_folder'])):
        os.makedirs(os.path.join(args['results_folder']))

    summary_path = os.path.join(args['results_folder'],'summaries',args['name'])
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)  

    checkpoint_path = os.path.join(args['results_folder'],'checkpoints', args['name'])
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    samples_path = os.path.join(args['results_folder'],'samples', args['name'])
    if not os.path.exists(samples_path):
        os.makedirs(samples_path)

    prediction_path = os.path.join(args['results_folder'],'predictions', args['name'])
    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)

    if not os.path.exists(downsample_directory):
        os.makedirs(downsample_directory)
    return args, reference_mesh_file, ds_factors, step_sizes, filter_sizes_enc, filter_sizes_dec, reference_points, summary_path, checkpoint_path, samples_path, prediction_path


def get_adj_trigs(A, F, reference_mesh):
    Adj = []
    for x in A:
        adj_x = []
        dx = x.todense()
        for i in range(x.shape[0]):
            adj_x.append(dx[i].nonzero()[1])
        Adj.append(adj_x)


    mesh_faces = reference_mesh.faces
    # Create Triangles List

    trigs_full = [[] for i in range(len(Adj[0]))]
    for t in mesh_faces:
        u, v, w = t
        trigs_full[u].append((u,v,w))
        trigs_full[v].append((u,v,w))
        trigs_full[w].append((u,v,w))

    Trigs = [trigs_full]
    for i,T in enumerate(F):
        trigs_down = [[] for i in range(len(Adj[i+1]))]
        for u,v,w in T:
            trigs_down[u].append((u,v,w))
            trigs_down[v].append((u,v,w))
            trigs_down[w].append((u,v,w))
        Trigs.append(trigs_down)
    
    return Adj, Trigs



def generate_spirals(step_sizes, M, Adj, Trigs, reference_points, dilation=None, random=False, counter_clockwise = True, nb_stds = 2):
    Adj_spirals = []
    for i in range(len(Adj)):
        mesh_vertices  = M[i].vertices
 
        sp = get_spirals(mesh_vertices, Adj[i],Trigs[i],reference_points[i], n_steps=step_sizes[i],\
                         padding='zero', counter_clockwise = counter_clockwise, random = random)
        Adj_spirals.append(sp)
        print('spiral generation for hierarchy %d (%d vertices) finished' %(i,len(Adj_spirals[-1])))

    
    ## Dilated convolution
    if dilation:
        for i in range(len(dilation)):
            dil = dilation[i]
            dil_spirals = []
            for j in range(len(Adj_spirals[i])):
                s = Adj_spirals[i][j][:1] + Adj_spirals[i][j][1::dil]
                dil_spirals.append(s)
            Adj_spirals[i] = dil_spirals
            
            
    # Calculate the lengths of spirals
    #   Use mean + 2 * std_dev, to capture 97% of data
    L = []
    for i in range(len(Adj_spirals)):
        L.append([])
        for j in range(len(Adj_spirals[i])):
            L[i].append(len(Adj_spirals[i][j]))
        L[i] = np.array(L[i])
    spiral_sizes = []
    for i in range(len(L)):
        sz = L[i].mean() + nb_stds*L[i].std()
        spiral_sizes.append(int(sz))
        print('spiral sizes for hierarchy %d:  %d' %(i,spiral_sizes[-1]))
    

    # 1) fill with -1 (index to the dummy vertex, i.e the zero padding) the spirals with length smaller than the chosen one
    # 2) Truncate larger spirals
    spirals_np = []
    for i in range(len(spiral_sizes)): #len(Adj_spirals)):
        S = np.zeros((1,len(Adj_spirals[i])+1,spiral_sizes[i])) - 1
        for j in range(len(Adj_spirals[i])):
            S[0,j,:len(Adj_spirals[i][j])] = Adj_spirals[i][j][:spiral_sizes[i]]
        #spirals_np.append(np.repeat(S,args['batch_size'],axis=0))
        spirals_np.append(S)

    return spirals_np, spiral_sizes, Adj_spirals





def distance(v,w):
    return np.sqrt(np.sum(np.square(v-w)))

def single_source_shortest_path(V,E,source,dist=None,prev=None):
    import heapq
    if dist == None:
        dist = [None for i in range(len(V))]
        prev = [None for i in range(len(V))]
    q = []
    seen = set()
    heapq.heappush(q,(0,source,None))
    while len(q) > 0 and len(seen) < len(V):
        d_,v,p = heapq.heappop(q)
        if v in seen:
            continue
        seen.add(v)
        prev[v] = p
        dist[v] = d_
        for w in E[v]:
            if w in seen:
                continue
            dw = d_ + distance(V[v],V[w])
            heapq.heappush(q,(dw,w,v))
    
    return prev, dist
        
        
    

def get_spirals(mesh, adj, trig, reference_points, n_steps=1, padding='zero', counter_clockwise = True, random = False):
    spirals = []
    
    if not random:
        heat_path = None
        dist = None
        for reference_point in reference_points:
            heat_path,dist = single_source_shortest_path(mesh,adj,reference_point, dist, heat_path)
        heat_source = reference_points

    for i in range(mesh.shape[0]):
        seen = set(); seen.add(i)
        trig_central = list(trig[i]); A = adj[i]; spiral = [i]
        
        # 1) Frist degree of freedom - choose starting pooint:
        if not random:
            if i in heat_source: # choose closest neighbor
                shortest_dist = np.inf
                init_vert = None
                for neighbor in A:
                    d = np.sum(np.square(mesh[i] - mesh[neighbor]))
                    if d < shortest_dist:
                        shortest_dist = d
                        init_vert = neighbor

            else: #   on the shortest path to the reference point
                init_vert = heat_path[i]
        else:
            # choose starting point:
            #   random for first ring
            init_vert = choice(A)
            
    
        
        
        # first ring
        if init_vert is not None:
            ring = [init_vert]; seen.add(init_vert)
        else:
            ring = []
        while len(trig_central) > 0 and init_vert is not None:
            cur_v = ring[-1]
            cur_t = [t for t in trig_central if t in trig[cur_v]]
            if len(ring) == 1:
                orientation_0 = (cur_t[0][0]==i and cur_t[0][1]==cur_v) \
                                or (cur_t[0][1]==i and cur_t[0][2]==cur_v) \
                                or (cur_t[0][2]==i and cur_t[0][0]==cur_v)
                if not counter_clockwise:
                    orientation_0 = not orientation_0
                    
                # 2) Second degree of freedom - 2nd point/orientation ambiguity
                if len(cur_t) >=2:
                # Choose the triangle that will direct the spiral counter-clockwise
                    if orientation_0:
                    # Third point in the triangle - next vertex in the spiral
                        third = [p for p in cur_t[0] if p!=i and p!=cur_v][0]
                        trig_central.remove(cur_t[0])
                    else:
                        third = [p for p in cur_t[1] if p!=i and p!=cur_v][0]
                        trig_central.remove(cur_t[1])
                    ring.append(third)
                    seen.add(third)      
                # 3) Stop if the spiral hits the boundary in the first point
                elif len(cur_t) == 1:
                    break
            else:
                # 4) Unique ordering for the rest of the points (3rd onwards) 
                if len(cur_t) >= 1:
                    # Third point in the triangle - next vertex in the spiral
                    third = [p for p in cur_t[0] if p!= cur_v and p!=i][0]   
                    # Don't append the spiral if the vertex has been visited already 
                    # (happens when the first ring is completed and the spiral returns to the central vertex)
                    if third not in seen:
                        ring.append(third)
                        seen.add(third)
                    trig_central.remove(cur_t[0])
            # 4) Stop when the spiral hits the boundary (the already visited triangle is no longer in the list): First half of the spiral
                elif len(cur_t) == 0:
                    break
                


                
      
        rev_i = len(ring)
        if init_vert is not None:
            v = init_vert

            if orientation_0 and len(ring)==1:
                reverse_order = False
            else:
                reverse_order = True
        need_padding = False  
        
        # 5) If on the boundary: restart from the initial vertex towards the other direction, 
        # but put the vertices in reverse order: Second half of the spiral
        # One exception if the starting point is on the boundary +  2nd point towards the desired direction
        while len(trig_central) > 0 and init_vert is not None:
            cur_t = [t for t in trig_central if t in trig[v]]
            if len(cur_t) != 1:
                break
            else:
                need_padding = True
                
            third = [p for p in cur_t[0] if p!=v and p!=i][0]
            trig_central.remove(cur_t[0])
            if third not in seen:
                ring.insert(rev_i,third)
                seen.add(third)
                if not reverse_order:
                    rev_i = len(ring)
                v = third
            
        # Add a dummy vertex between the first half of the spiral and the second half - similar to zero padding in a 2d grid
        if need_padding:
            ring.insert(rev_i,-1)
            """
            ring_copy = list(ring[1:])
            rev_i = rev_i - 1
            for z in range(len(ring_copy)-2):
                if padding == 'zero':
                    ring.insert(rev_i,-1) # -1 is our sink node
                elif padding == 'mirror':
                    ring.insert(rev_i,ring_copy[rev_i-z-1])
            """
        spiral += ring
        
        
        # Next rings:
        for step in range(n_steps-1):
            next_ring = set([]); next_trigs = set([]); 
            if len(ring) == 0:
                break
            base_triangle = None
            init_vert = None
            
            # Find next hop neighbors
            for w in ring:
                if w!=-1:
                    for u in adj[w]:
                        if u not in seen:
                            next_ring.add(u)
            
            # Find triangles that contain two outer ring nodes. That way one can folllow the spiral ordering in the same way 
            # as done in the first ring: by simply discarding the already visited triangles+nodes.
            for u in next_ring:
                for tr in trig[u]:
                    if len([x for x in tr if x in seen]) == 1:
                        next_trigs.add(tr)
                    elif ring[0] in tr and ring[-1] in tr:
                        base_triangle = tr
            # Normal case: starting point in the second ring -> 
            # the 3rd point in the triangle that connects the 1st and the last point in the 1st ring with the 2nd ring
            if base_triangle is not None:
                init_vert = [x for x in base_triangle if x != ring[0] and x != ring[-1]]
                # Make sure that the the initial point is appropriate for starting the spiral,
                # i.e it is connected to at least one of the next candidate vertices
                if len(list(next_trigs.intersection(set(trig[init_vert[0]]))))==0:
                    init_vert = None


            # If no such triangle exists (one of the vertices is dummy, 
            # or both the first and the last vertex take part in a specific type of boundary)
            # or the init vertex is not connected with the rest of the ring -->
            # Find the relative point in the the triangle that connects the 1st point with the 2nd, or the 2nd with the 3rd
            # and so on and so forth. Note: This is a slight abuse of the spiral topology
            if init_vert is None:
                for r in range(len(ring)-1):
                    if ring[r] !=-1 and ring[r+1]!=-1:
                        tr = [t for t in trig[ring[r]] if t in trig[ring[r+1]]]
                        for t in tr:
                            init_vert = [v for v in t if v not in seen] 
                            # make sure that the next vertex is appropriate to start the spiral ordering in the next ring
                            if len(init_vert)>0 and len(list(next_trigs.intersection(set(trig[init_vert[0]]))))>0:
                                break
                            else:
                                init_vert = []
                        if len(init_vert)>0  and len(list(next_trigs.intersection(set(trig[init_vert[0]]))))>0:
                            break
                        else:
                            init_vert = []

            
            # The rest of the procedure is the same as the first ring
            if init_vert is None:
                init_vert = []
            if len(init_vert)>0:
                init_vert = init_vert[0]
                ring = [init_vert]
                seen.add(init_vert)
            else:
                init_vert = None
                ring = []

#             if i == 57:
#                 import pdb;pdb.set_trace()
            while len(next_trigs) > 0 and init_vert is not None:
                cur_v = ring[-1]
                cur_t = list(next_trigs.intersection(set(trig[cur_v])))
                    
                if len(ring) == 1:
                    try:
                        orientation_0 = (cur_t[0][0] in seen and cur_t[0][1]==cur_v) \
                                        or (cur_t[0][1] in seen and cur_t[0][2]==cur_v) \
                                        or (cur_t[0][2] in seen and cur_t[0][0]==cur_v)
                    except:
                        import pdb;pdb.set_trace()
                    if not counter_clockwise:
                        orientation_0 = not orientation_0

                    # 1) orientation ambiguity for the next ring
                    if len(cur_t) >=2:
                    # Choose the triangle that will direct the spiral counter-clockwise
                        if orientation_0:
                        # Third point in the triangle - next vertex in the spiral
                            third = [p for p in cur_t[0] if p not in seen and p!=cur_v][0]
                            next_trigs.remove(cur_t[0])
                        else:
                            third = [p for p in cur_t[1] if p not in seen and p!=cur_v][0]
                            next_trigs.remove(cur_t[1])
                        ring.append(third)
                        seen.add(third)      
                    # 2) Stop if the spiral hits the boundary in the first point
                    elif len(cur_t) == 1:
                        break
                else:
                    # 3) Unique ordering for the rest of the points
                    if len(cur_t) >= 1:
                        third = [p for p in cur_t[0] if p != v and p not in seen]
                        next_trigs.remove(cur_t[0])
                        if len(third)>0:
                            third = third[0]
                            if third not in seen:
                                ring.append(third)
                                seen.add(third)
                        else:
                            break
                    # 4) Stop when the spiral hits the boundary 
                    # (the already visited triangle is no longer in the list): First half of the spiral
                    elif len(cur_t) == 0:
                        break
                    
                       
            rev_i = len(ring)
            if init_vert is not None:
                v = init_vert

                if orientation_0 and len(ring)==1:
                    reverse_order = False
                else:
                    reverse_order = True
        
            need_padding = False
            
            while len(next_trigs) > 0 and init_vert is not None:
                cur_t = [t for t in next_trigs if t in trig[v]]
                if len(cur_t) != 1:
                    break
                else:
                    need_padding = True
                    
                third = [p for p in cur_t[0] if p!=v and p not in seen]
                next_trigs.remove(cur_t[0])
                if len(third)>0:
                    third = third[0]
                    if third not in seen:
                        ring.insert(rev_i,third)
                        seen.add(third)
                    if not reverse_order:
                        rev_i = len(ring)
                    v = third
            
            if need_padding:
                ring.insert(rev_i,-1)
                """
                ring_copy = list(ring[1:])
                rev_i = rev_i - 1
                for z in range(len(ring_copy)-2):
                    if padding == 'zero':
                        ring.insert(rev_i,-1) # -1 is our sink node
                    elif padding == 'mirror':
                        ring.insert(rev_i,ring_copy[rev_i-z-1])
                """

            spiral += ring  
        
        spirals.append(spiral)
    return spirals
