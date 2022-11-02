import pandas as pd
import numpy as np
from collections import defaultdict
from collections import Counter
import networkx as nx
from matplotlib import pyplot as plt

def import_facebook_data(file):
    E = np.loadtxt(file, dtype=int)
    E=np.unique(E, axis=0)
    return E

def import_bitcoin_data(path):


    data = pd.read_csv(path)
    df1 = data.iloc[:,0:2] 
    E=df1.to_numpy()
    E=np.subtract(E,1)
    E=np.unique(E,axis=0)
    return E

def spectralDecomp_OneIter(edge_list):

    size=edge_list.max()+1
    adjacency = [[0]*size for _ in range(size)]
    for sink, source in edge_list:
        adjacency[sink][source] = 1
        adjacency[source][sink]=1

    A=np.array(adjacency)


    size=len(A)
    nodes=np.arange(start=0,stop=size,step=1)
    temp=np.ones((size,1))
    diag=np.matmul(A,temp).T[0]
    D=np.diag(diag)
    L=D-A
    eig_val,eig_vec=np.linalg.eigh(L) # Since symmetric


    eig_val=np.real(eig_val)
    eig_vec=np.real(eig_vec)
    intm=np.sort(eig_val)

    part_index=0
    while intm[part_index]<pow(.5,5): # Note that number of partition chnages with this threshold
        part_index+=1

    index = np.argsort(eig_val)[part_index]
    fielder_vec= eig_vec[index]  

    partition = -np.ones(len(A)) # Follows this structure because recursion follows this

    if(np.all(fielder_vec>=0)):# Edge case 1
        
        partition[np.intersect1d(np.where(fielder_vec>=0)[0],nodes)] = edge_list.min()
        graph_partition=[]
        graph_partition.append(nodes)
        graph_partition.append(partition)
        graph_partition=np.array(graph_partition).T
        return fielder_vec,A,partition


    elif(np.all(fielder_vec<0)): # edge case 2
        partition[np.intersect1d(np.where(fielder_vec<0)[0],nodes)] = edge_list.min()
        graph_partition=[]
        graph_partition.append(nodes)
        graph_partition.append(partition)
        graph_partition=np.array(graph_partition).T
        return fielder_vec,A,partition



    pos_id=np.where(fielder_vec>=0)[0].min()
    neg_id=np.where(fielder_vec<0)[0].min()
    operation = lambda x : pos_id if(x >= 0) else neg_id
    partition = np.array(list(map(operation,fielder_vec)))
    graph_partition=[]
    graph_partition.append(nodes)
    graph_partition.append(partition)
    graph_partition=np.array(graph_partition).T
    return fielder_vec, A, partition

def louvain_one_iter(nodes_connectivity_list_fb):
    def get_adj(nodes_connectivity_list_fb):
        size=nodes_connectivity_list_fb.max()+1
        # size = len(set([n for e in nodes_connectivity_list_fb for n in e])) 
        # # make an empty adjacency list  
        adjacency = [[0]*size for _ in range(size)]
        # populate the list for each edge
        for sink, source in nodes_connectivity_list_fb:
            adjacency[sink][source] = 1
            adjacency[source][sink]=1
        return np.array(adjacency)
    def get_parameters():
        A= get_adj(nodes_connectivity_list_fb)
        deg=np.sum(A, axis=0)
        node_to_commun_mapping =dict()
        degree_of_community=dict()
        degrees=dict()
        N=len(A)
        m=np.sum(A)
        nodes=np.arange(start=0,stop=N,step=1)

        Neigh = defaultdict(list)
        for i in range(N):
            flag=False
            for j in range(N):
                if A[i][j]==1:
                    Neigh.update({i:Neigh[i]+[j]})
                    flag=True
                else:
                    pass
            if flag==False:
                Neigh.update({i:[]})
    

        for i in nodes:
            node_to_commun_mapping[i] = i
            degrees[i] =deg[i]
            degree_of_community[i]=deg[i]
        return node_to_commun_mapping,degree_of_community,degrees,m,nodes,Neigh
    # Initial Condition


    node_to_commun_mapping,degree_of_community,degree_of_node,m,nodes,Neigh=get_parameters()

    flag=True

    while flag==True:
        flag = False

        for i in nodes:
            i_s_community = node_to_commun_mapping[i]
            Neighbours=Neigh[i]
            Neighbours_community=[]
            for x in Neighbours:
                Neighbours_community.append(node_to_commun_mapping[x])
            temp_dict=dict(Counter(Neighbours_community))
            k_i_in_list=[]
            for x in Neighbours_community:
                k_i_in_list.append(temp_dict[x])
            
            
            
            # Preparing 
            degree_of_community[i_s_community] = degree_of_community[i_s_community]-degree_of_node[i]
            prev=node_to_commun_mapping[i]
            node_to_commun_mapping[i] = -1

            #setting init values
            opt_delta_Q = 0
            opt_comm=-2

            for neigh_commun, k_i_in_p in zip(Neighbours_community,k_i_in_list):
                delta_Q = 2 * k_i_in_p - degree_of_community[neigh_commun] * degree_of_node[i]/ m
                if delta_Q > opt_delta_Q:
                    opt_delta_Q = delta_Q
                    opt_comm=neigh_commun
            if opt_comm==-2:
                node_to_commun_mapping[i]=prev
                continue
            

            # update info

 
            degree_of_community[opt_comm] =degree_of_community[opt_comm]+ degree_of_node[i]
            node_to_commun_mapping[i] = opt_comm
            
            if opt_comm != i_s_community:
                flag = True


    
    classes=list(set(node_to_commun_mapping.values()))

    for x in classes:
        m=[k for k,v in node_to_commun_mapping.items() if v == x]
        id=min(m)
        for x in m:
            node_to_commun_mapping.update({x:id})

    


    graph_partition_vector=[]
    graph_partition_vector.append(list(node_to_commun_mapping.keys()))
    graph_partition_vector.append(list(node_to_commun_mapping.values()))
    graph_partition_vector=np.array(graph_partition_vector).T


    return graph_partition_vector

def Rec_Dcmps(edge_list, division, transform):
    def get_bin_partioned(edge_list):
        fielder_vec, _, _ = spectralDecomp_OneIter(edge_list)

        comm_neg= np.where(fielder_vec<0)[0]
        comm_pos = np.where(fielder_vec>=0)[0]

        return comm_neg,comm_pos
    def modify_partiton(pos,partition,nodelist):

        min=100000   
        comm_pos_index=[]                       # Assigning min node number as the cluster id
        for i in nodelist[pos]:
            comm_pos_index.append(inverse_transform[i])
            if inverse_transform[i]<min:
                min=inverse_transform[i]
        comm_pos_index=np.array(comm_pos_index)

        partition[comm_pos_index,1]=min

        return partition
    def get_newEdgeList(souce_nodes,dest_nodes,map):
        nonlocal inverse_transform
        edge_list_zero=[]
        for x,y in zip(souce_nodes,dest_nodes):
            edge_list_zero.append([map[inverse_transform[x]],map[inverse_transform[y]]])
        edge_list_zero=np.array(edge_list_zero)
        return edge_list_zero

    check_list=division[:,1]
    stand_list=np.ones(len(check_list))*-1

    print("check")
    if (check_list!=stand_list).all():
        return division


    inverse_transform=dict()
    for i in range(len(transform.values())):
        inverse_transform[list(transform.values())[i]]=list(transform.keys())[i]


    comm_neg,comm_pos=get_bin_partioned(edge_list)
    node_list = np.array(list(transform.values()))


    ln=len(comm_neg)
    lp=len(comm_pos)


   
    if(ln==0 or lp==0): # No more recursion
        if ln==0:
            pos=comm_pos
        else:
            pos=comm_neg        
        return modify_partiton(pos,division,node_list)


    map_pos=dict()
    map_neg=dict()

    pos_map_one=np.arange(lp)
    neg_map_one=np.arange(ln)

    l=len(node_list[comm_pos])

    for i in range(l):
        map_pos.update({inverse_transform[node_list[comm_pos][i]]:pos_map_one[i]})
    l=len(node_list[comm_neg])   
    for i in range(l):
        map_neg.update({inverse_transform[node_list[comm_neg][i]]:neg_map_one[i]})
    

    new_source=[]
    new_dest=[]
    new_source_bar=[]
    new_dest_bar=[]

   
    for i in range(len(edge_list)):
        flag1=(edge_list[i][0] in comm_neg) and (edge_list[i][1] in comm_neg)
        flag2= (edge_list[i][0] in comm_pos) and ( edge_list[i][1] in comm_pos)
        if flag1:
            new_source.append(edge_list[i][0])
            new_dest.append(edge_list[i][1])
        if flag2:
            new_source_bar.append(edge_list[i][0])
            new_dest_bar.append(edge_list[i][1])

    

    edge_list_neg=get_newEdgeList(new_source,new_dest,map_neg)
    edge_list_pos=get_newEdgeList(new_source_bar,new_dest_bar,map_pos)
    l_n=len(edge_list_neg)
    l_p=len(edge_list_pos)


    if(l_n>0):
        Rec_Dcmps(edge_list_neg,division, map_neg)
    if(l_p>0):
        Rec_Dcmps(edge_list_pos,division, map_pos)
    return division

def spectralDecomposition(edge_list):
    def get_init_param(edge_list):
        size=edge_list.max()+1
        node_ids=np.arange(start=0,stop=size,step=1)
        init_pos=-1*np.ones_like(node_ids)
        partition=np.vstack([node_ids,init_pos]).T
        transform=dict()
        for i in node_ids:
            transform.update({i:i})     # each in it's own community
        return partition,transform
    def get_adj(nodes_connectivity_list_fb):
        size=nodes_connectivity_list_fb.max()+1
        # size = len(set([n for e in nodes_connectivity_list_fb for n in e])) 
        # # make an empty adjacency list  
        adjacency = [[0]*size for _ in range(size)]
        # populate the list for each edge
        for sink, source in nodes_connectivity_list_fb:
            adjacency[sink][source] = 1
            adjacency[source][sink]=1
        return np.array(adjacency)
    
    A = get_adj(edge_list)
    partition,map=get_init_param(edge_list)

    partition = Rec_Dcmps(edge_list,partition,map)
    positions=np.where(partition[:,1]==-1)[0]

    for i in positions:
        freq_dict=dict()
        pos=np.where(A[i]==1)[0]
        temp=[]
        for x in pos:
            temp.append(partition[:,1][x])
        templist=temp
        temp=np.array(temp)
        labels=np.unique(temp)
        for x in labels:
            freq_dict.update({x:templist.count(x)})
        unique=labels
        counts=np.array(list(freq_dict.values()))

        l_u=len(unique)
        l_c=len(counts)
        pos_sorted_count=np.argsort(counts)
        if(l_u==0):
            value=-1
        elif(unique[pos_sorted_count[l_c-1]] == -1):
            if(l_u>1):
                value=unique[pos_sorted_count[l_c-2]]
            else:
                # pass
                continue
        else:
            value=unique[pos_sorted_count[l_c-1]]
        partition[i,1] =value

    while(len(np.where(partition[:,0]<partition[:,1])[0])):
        idx = np.where(partition[:,0]<partition[:,1])[0][0]
        partition[np.where(partition[:,1]==partition[idx,1]),1] = idx
    
    return partition

def createSortedAdjMat(partition, edge_list):
    def get_adj(nodes_connectivity_list_fb):
        size=nodes_connectivity_list_fb.max()+1
        # size = len(set([n for e in nodes_connectivity_list_fb for n in e])) 
        # # make an empty adjacency list  
        adjacency = [[0]*size for _ in range(size)]
        # populate the list for each edge
        for sink, source in nodes_connectivity_list_fb:
            adjacency[sink][source] = 1
            adjacency[source][sink]=1
        return np.array(adjacency)
    def get_clustered_graph(G):
        nonlocal partition
        clusted_graph=dict()
        nodes=np.argsort(partition[:,1])
        for x in range(len(nodes)):
            clusted_graph.update({nodes[x]:partition[:,0][x]})
        H = nx.relabel_nodes(G, clusted_graph)
        return H

    G=nx.from_numpy_matrix(get_adj(edge_list))
    H=get_clustered_graph(G)
    A_bar=nx.to_numpy_matrix(H)
    return A_bar





if __name__ == "__main__":

    ########### Answer qn 1-4 for facebook data #################################################
    # Import facebook_combined.txt
    # nodes_connectivity_list is a nx2 numpy array, where every row 
    # is a edge connecting i<->j (entry in the first column is node i, 
    # entry in the second column is node j)
    # Each row represents a unique edge. Hence, any repetitions in data must be cleaned away.

    
    nodes_connectivity_list_fb = import_facebook_data("./data/facebook_combined.txt")


    # This is for question no. 1
    # fielder_vec    : n-length numpy array. (n being number of nodes in the network)
    # adj_mat        : nxn adjacency matrix of the graph
    # graph_partition: graph_partitition is a nx2 numpy array where the first column consists of all
    #                  nodes in the network and the second column lists their community id (starting from 0)
    #                  Follow the convention that the community id is equal to the lowest nodeID in that community.
    fielder_vec_fb, adj_mat_fb, graph_partition_fb = spectralDecomp_OneIter(nodes_connectivity_list_fb)
 

    # This is for question no. 2. Use the function 
    # written for question no.1 iteratetively within this function.
    # graph_partition is a nx2 numpy array, as before. It now contains all the community id's that you have
    # identified as part of question 2. The naming convention for the community id is as before.
    
    graph_partition_fb = spectralDecomposition(nodes_connectivity_list_fb)
    

    # This is for question no. 3
    # Create the sorted adjacency matrix of the entire graph. You will need the identified communities from
    # question 3 (in the form of the nx2 numpy array graph_partition) and the nodes_connectivity_list. The
    # adjacency matrix is to be sorted in an increasing order of communitites.
    
    clustered_adj_mat_fb = createSortedAdjMat(graph_partition_fb, nodes_connectivity_list_fb)

    # This is for question no. 4
    # run one iteration of louvain algorithm and return the resulting graph_partition. The description of
    # graph_partition vector is as before.
    
    graph_partition_louvain_fb = louvain_one_iter(nodes_connectivity_list_fb)
    # print(graph_partition_louvain_fb)


    ############ Answer qn 1-4 for bitcoin data #################################################
    # Import soc-sign-bitcoinotc.csv
    
    nodes_connectivity_list_btc = import_bitcoin_data("./data/soc-sign-bitcoinotc.csv")
    # nodes_connectivity_list_btc = import_bitcoin_data("./data/graph.csv")


    # Question 1
    # fielder_vec_btc, adj_mat_btc, graph_partition_btc = spectralDecomp_OneIter(nodes_connectivity_list_btc)

    # Question 2
    graph_partition_btc = spectralDecomposition(nodes_connectivity_list_btc)


    # Question 3
    clustered_adj_mat_btc = createSortedAdjMat(graph_partition_btc, nodes_connectivity_list_btc)

    # Question 4
    graph_partition_louvain_btc = louvain_one_iter(nodes_connectivity_list_btc)