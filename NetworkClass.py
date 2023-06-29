# Packages
import networkx as nx
import community
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyvis
from pyvis.network import Network  
import math
import copy
from cdlib import algorithms

class NetworkClass:
    """
    Class to build and visualize a network starting from edge data.
        
    Methods
    ----------
    add_note_attr: adds an attribute the network.
    network_statistics: computes descriptive statistics of the network.
    communities: shows communities of the network (Louvain method) and eventually colors nodes according to their community.
    """

    # We need to define a nx network and a corresponding Pyvis network (with attributes for coloring if needed)
    def __init__(self, data, node_col, neighbor_col, edge_weight_col=None, dropna_viz=False, base_color = '#ACACAC', base_node_size = None, base_edge_width = None, int_id = False, digraph=False, drop_edges_viz=False,labels=True,font_size = 100):
        """
        Inputs when instantiating
        ----------
        data : pd.DataFrame
            dataframe of connections (in the form node, weight of link, neighbor)
        node_col : str
            node column name
        neighbor_col : str
            neighbor column name
        edge_weight_col : str
            edge weight column name
        dropna_viz : bool
            drop singletons from pyvis visualization (but not from nx graph)
        base_color: str
            hex default color for nodes and edges
        base_node_size: int
            default node size
        base_edge_width: int
            default edge width
        int_id: bool
            set to True when your node_neighbor id is numeric and you want to make sure that it is converted to int
        labels : bool (default True)
            if set to False, remove labels from edges
        font_size: int
            font sizes for the labels (default 100)
        """
        
        self.data = data
        
    ###### NETWORKX ASSIGNMENT ######      
        if int_id:
            data[[node_col,neighbor_col]] = data[[node_col,neighbor_col]].astype(int)
        
        if digraph:
            self.g = nx.DiGraph()
        else:
            self.g = nx.Graph()
            
        self.g.add_nodes_from([str(el) for el in data[node_col].unique()])
        if edge_weight_col != None: # We may not always have weighted edges
            elist = list(zip(data[node_col],data[neighbor_col],data[edge_weight_col]))
            self.g.add_weighted_edges_from([(str(el[0]),str(el[1]),el[2]) for el in elist])
        else:
            elist = list(zip(data[node_col],data[neighbor_col]))
            self.g.add_edges_from([(str(el[0]),str(el[1])) for el in elist])
        
    ###### PYVIS ASSIGNMENT ######
        self.g_copy = copy.deepcopy(self.g)
        
        # Drop edges below a certain threshold for visualization
        if drop_edges_viz:
            for u,v,d in self.g.edges(data=True):
                if d['weight'] <= drop_edges_viz:
                    self.g_copy.remove_edge(u, v)
        
        # Rescale edges 
        for u,v,d in self.g_copy.edges(data=True):
            d['weight'] = np.sqrt(d['weight'])

        self.nt = Network('1500px','1500px',directed=digraph, bgcolor="#222222", font_color="white") #,select_menu=True,filter_menu=True)
        self.nt.barnes_hut()
        self.nt.from_nx(self.g_copy) 
        
        # base (default) graphical settings
        for node in self.nt.nodes:
            node['color'] = base_color
            if base_node_size != None:
                node['size'] = base_node_size
        for edge in self.nt.edges:
            if base_edge_width != None:
                edge['width'] = base_edge_width
                
        # set graphical settings for nodes
        for node in self.nt.nodes:
            node['title'] = node['label'] # titles are what appear when you hover over a node
            node['font']['size'] = font_size
            node['label'] = node['label'] + ' (' + str(self.g_copy.degree(node['id'])) + ')'
            if not labels:
                del node['label']  
        
        # Enable physics
        self.nt.toggle_physics(True)
        
        
    def add_node_attr(self,data,attr_name,key='id'): # we can add attributes via a DataFrame or a dictionary
        """
        Method to add an attribute the network.

        Arguments
        ----------
        data : pd.DataFrame/dict
            dataframe/dictionary containing a mapping between node IDs and the characteristic to be added. Node IDs (key for the mapping) must be in the index
        attr_name : str
            name of the attribute to be added (must be a column name if "data" is a dataframe)
        """
        
        if isinstance(data,pd.DataFrame):
            for node in self.nt.nodes:
                node_key = node[key]
                try: # not all nodes may be in the mapping dataframe/dictionary
                    node[attr_name] = data.loc[node_key,attr_name]
                except:
                    pass
        if isinstance(data,dict):
            for node in self.nt.nodes:
                node_key = node[key]
                try:
                    node[attr_name] = data[node_key]
                except:
                    pass 
                
    def network_statistics(self,name='Network statistics.xlsx',directed=False):
        """
        Method to compute descriptive statistics of the network.

        Arguments
        ----------     
        name : str
            Name of Excel file with network metrics (with path).
        directed: bool
            Whether the network is directed or not (default False).
        """
        
        if directed:
            out_degree_sequence = sorted((d for n, d in self.g.out_degree()), reverse=True)
            avg_degree = sum(out_degree_sequence)/len(out_degree_sequence)
        else:
            degree_sequence = sorted((d for n, d in self.g.degree()), reverse=True)
            avg_degree = sum(degree_sequence)/len(degree_sequence)
            
        self.g_und = self.g.to_undirected()
        self.partition = community.best_partition(self.g_und)
        
        self.stats = pd.DataFrame(columns=['Statistic', 'Value'])
        self.stats.loc[0] = ['1. Number of nodes',self.g.number_of_nodes()]
        self.stats.loc[1] = ['2. Number of edges',self.g.number_of_edges()]
        self.stats.loc[2] = ['3. Average degree',avg_degree]
        self.stats.loc[3] = ['4. Density',nx.density(self.g)]
        self.stats.loc[4] = ['5. Average clustering',nx.average_clustering(self.g)]
        self.stats.loc[6] = ['6. Modularity',community.modularity(self.partition,self.g_und)]
        
        self.stats.to_excel(name)
        
    def communities(self,palette=None):
        """
        Method to show communities of the network (Louvain method) and eventually color nodes according to their community.

        Arguments
        ----------
        palette : dict
            dictionary with color palette (color code: hex value)
        """
        
        self.g_und = self.g.to_undirected()
        self.partition = algorithms.louvain(self.g_und, resolution=1., randomize=False)
        print(self.partition.communities) # identified clustering
        
        if palette != None: # color nodes based on community
            self.color_dict = {}
            for community,color_code in zip(self.partition.communities,palette.keys()):
                for element in community:
                    self.color_dict[element] = palette[color_code]
                    
            self.add_node_attr(data=self.color_dict,attr_name='color')