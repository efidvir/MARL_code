import re

with open("c:/Users/efid/OneDrive - Ceragon/UNITY-6G/WP4/MARL_code/sixg_sim/simulation.py", "r", encoding="utf-8") as f:
    content = f.read()

new_forward_traffic = '''    def _forward_traffic(self, traffic_arrivals):
        """
        Forward traffic through the network.
        Now uses proper physical path traversal across Oran infra, checking Link capacity
        and hitting the MARL agent transit policies at EVERY hop (RU->DU->CU).
        """
        import networkx as nx
        from .topology import TrafficClass
        
        self.ue_to_ue_success_count = 0
        self.ue_to_ue_failure_count = 0
        ue_to_ue_volume = 0.0
        ue_to_ue_offered = 0.0
        ue_general_volume = 0.0
        ue_general_offered = 0.0
        
        ue_to_ue_participants = set()
        for s, t in self.ue_to_ue_flows:
            ue_to_ue_participants.add(s)
            ue_to_ue_participants.add(t)
            
        def apply_marl_policy(path_nodes, traffic_class, amount):
            bottleneck = amount
            for n_id in path_nodes:
                agent = self.agents.get(n_id)
                if agent and hasattr(agent, 'last_actions') and traffic_class in agent.last_actions:
                    mode = agent.last_actions[traffic_class].get('admission_mode', 'ADMIT').upper()
                    if mode == 'THROTTLE': bottleneck = min(bottleneck, amount * 0.5)
                    elif mode == 'HOLD': return 0.0
            return bottleneck

        def consume_path(path_nodes, amount):
            final_bottleneck = amount
            for i in range(len(path_nodes)-1):
                u, v = path_nodes[i], path_nodes[i+1]
                link_data = self.topology.graph.get_edge_data(u, v)
                if link_data and 'link_id' in link_data:
                    link_id = link_data['link_id']
                    if getattr(self.topology.links[link_id], 'is_up', True):
                        final_bottleneck = min(final_bottleneck, self.topology.links[link_id].available_capacity())
                    else:
                        return 0.0
            
            if final_bottleneck <= 0: return 0.0
            
            for i in range(len(path_nodes)-1):
                u, v = path_nodes[i], path_nodes[i+1]
                link_data = self.topology.graph.get_edge_data(u, v)
                if link_data and 'link_id' in link_data:
                    self.topology.links[link_data['link_id']].add_traffic(final_bottleneck)
            
            for n_id in path_nodes:
                if n_id in self.topology.nodes:
                    self.topology.nodes[n_id].update_energy(final_bottleneck, 0)
                    
            return final_bottleneck

        # 1. Process UE-to-UE flows
        for source_ue, target_ue in self.ue_to_ue_flows:
            source_node = self.topology.nodes.get(source_ue)
            target_node = self.topology.nodes.get(target_ue)

            if not source_node or not source_node.is_survivor: continue
            if not target_node or not target_node.is_survivor:
                self.ue_to_ue_failure_count += 1
                continue

            base_ue_to_ue_traffic = 0.0 if self.island_mode else 8.0
            emergency_traffic = 0.0
            if source_node.emergency_state:
                if getattr(source_node, 'emergency_type', '') == 'emergency': emergency_traffic = 15.0
                elif getattr(source_node, 'emergency_type', '') == 'imminent_peril': emergency_traffic = 20.0
                else: emergency_traffic = 10.0
            rescue_bonus = 12.0 if (source_node.is_rescue_service or target_node.is_rescue_service) else 0.0

            ue_to_ue_traffic = base_ue_to_ue_traffic + emergency_traffic + rescue_bonus
            ue_to_ue_offered += ue_to_ue_traffic
            
            if self._can_route_ue_to_ue(source_ue, target_ue):
                try:
                    path = nx.shortest_path(self.topology.graph, source_ue, target_ue)
                    admitted = apply_marl_policy(path, TrafficClass.LIFE_SAFETY, ue_to_ue_traffic)
                    delivered = consume_path(path, admitted)
                    
                    if delivered > 0:
                        self.ue_to_ue_success_count += 1
                        ue_to_ue_volume += delivered
                    else:
                        self.ue_to_ue_failure_count += 1
                except nx.NetworkXNoPath:
                    self.ue_to_ue_failure_count += 1
            else:
                self.ue_to_ue_failure_count += 1

        if self.island_mode and self.marl_ue_routing_enabled and self.current_tick % 20 == 0:
            print(f"[ISLAND] t={self.current_tick}: UE-to-UE traffic - {self.ue_to_ue_success_count}/{len(self.ue_to_ue_flows)} flows, volume: {ue_to_ue_volume:.1f} units")
        
        # 2. General UE Traffic
        core_targets = [n for n in getattr(self, 'core_nodes', []) if self.topology.nodes.get(n) and self.topology.nodes[n].is_survivor]
        
        for node_id, arrivals in traffic_arrivals.items():
            if node_id not in self.topology.nodes: continue
            node = self.topology.nodes[node_id]
            
            if getattr(node.node_type, 'value', str(node.node_type)) == "UE" and node.is_survivor:
                ue_to_ue_amount = 0.0
                if not self.island_mode and node_id in ue_to_ue_participants:
                    flows_cnt = sum(1 for s, t in self.ue_to_ue_flows if (s == node_id or t == node_id) and getattr(self.topology.nodes.get(s), 'is_survivor', False) and getattr(self.topology.nodes.get(t), 'is_survivor', False))
                    ue_to_ue_amount = flows_cnt * 1.5
                
                clean_arrivals = {}
                for tc, amount in arrivals.items():
                    gen_amount = max(0, amount - ue_to_ue_amount) if (tc == TrafficClass.LIFE_SAFETY and node_id in ue_to_ue_participants) else amount
                    clean_arrivals[tc] = gen_amount
                    ue_general_offered += gen_amount
                
                best_path = None
                if not self.island_mode and core_targets:
                    paths = []
                    for c in core_targets:
                        if self.topology.has_path(node_id, c):
                            paths.append(nx.shortest_path(self.topology.graph, node_id, c))
                    if paths:
                        best_path = min(paths, key=len)
                
                for tc, amount in clean_arrivals.items():
                    queue = node.queues[tc]
                    queue.offered_load = amount
                    delivered = 0.0
                    if best_path and amount > 0:
                        admitted = apply_marl_policy(best_path, tc, amount)
                        delivered = consume_path(best_path, admitted)
                        
                    queue.admitted_load = delivered
                    queue.queued_load += delivered
                    queue.dropped_load = amount - delivered
                    queue.delivered_load = delivered
                    
                    if tc != TrafficClass.LIFE_SAFETY or node_id not in ue_to_ue_participants:
                        ue_general_volume += delivered
            else:
                total_traffic = sum(arrivals.values())
                node.update_energy(total_traffic, 0)
                for tc, amount in arrivals.items():
                    q = node.queues[tc]
                    q.offered_load = amount
                    q.admitted_load = amount
                    q.delivered_load = amount
                    q.dropped_load = 0

        self.last_tick_ue_to_ue_volume = ue_to_ue_volume
        self.last_tick_ue_general_volume = ue_general_volume
        self.last_tick_ue_to_ue_offered = ue_to_ue_offered
        self.last_tick_ue_general_offered = ue_general_offered'''

# Use regex to replace the function entirely
# We find def _forward_traffic(self, traffic_arrivals... and replace until def _collect_metrics(self):
pattern = re.compile(r'    def _forward_traffic\(self, traffic_arrivals: Dict\[str, Dict\[TrafficClass, float\]\]\):.*?(?=(    def _collect_metrics\(self\):))', re.DOTALL)

if pattern.search(content):
    new_content = pattern.sub(new_forward_traffic + "\n", content)
    with open("c:/Users/efid/OneDrive - Ceragon/UNITY-6G/WP4/MARL_code/sixg_sim/simulation.py", "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Successfully replaced _forward_traffic.")
else:
    print("Could not find _forward_traffic to replace.")
