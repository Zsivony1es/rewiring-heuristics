class GraphFactory:
    """Used to generate a single graph, based on set parameters."""

    def __init__(self, json_conf=None) -> None:
        if (json_conf != None):
            self._get_values_from_json(json_conf=json_conf)
        else:
            self._set_null_attributes()
        
            
    def _set_null_attributes(self) -> None:
        self.edge_count = None
        self.edge_attr_count = None
        self.node_count = None
        self.node_attr_count = None


    def _get_values_from_json(self, json_conf) -> None:
        """Used to set"""
        pass

    def generate_graph(self, path) -> None:
        """Generates the graph data into the directory specified by `path`."""
        pass


