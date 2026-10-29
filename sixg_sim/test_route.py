import sys
sys.path.append("..")
from sixg_sim.topology import load_topology_from_yaml
import networkx as nx

topology = load_topology_from_yaml("../config/topology_example.yaml")
path_nodes = ['UE1', 'gNB1', 'Rescue_UE_1']
amount = 15.0
final_bottleneck = amount

print("Testing consume_path logic:")
for i in range(len(path_nodes)-1):
    u, v = path_nodes[i], path_nodes[i+1]
    link_data = topology.graph.get_edge_data(u, v)
    print(f"Edge {u}->{v}: link_data={link_data}")
    if link_data and 'link_id' in link_data:
        link_id = link_data['link_id']
        link = topology.links[link_id]
        print(f"  Link {link_id}: is_up={getattr(link, 'is_up', True)}, available={link.available_capacity()}")
        if getattr(link, 'is_up', True):
            final_bottleneck = min(final_bottleneck, link.available_capacity())
        else:
            final_bottleneck = 0.0
            break
            
print(f"Final bottleneck: {final_bottleneck}")
