import igraph as ig

class DataPreprocessor(object):
    # test ok
    def __init__(self, node_sequence_size, receptive_field_size, num_channels):
        self.node_sequence_size = node_sequence_size
        self.receptive_field_size = receptive_field_size
        self.channels = num_channels
    def execute(self, graph):
        self.channels.set_graph(graph)

        # calculate centrality and make it to be vertex properties
        bv = graph.close
        graph.vp['betweenness'] = graph.new_vertex_property('double')
        graph.vp.betweenness = bv

        # decide a node sequence
        num_vertices = graph.num_vertices()
        node_sequence = sorted(
            list(graph.vertices()),
            key=lambda x: bv[x],
            reverse=True)[: min(num_vertices, self.node_sequence_size)]

        receptive_field_maker = ReceptiveFieldMaker(self.receptive_field_size)
        receptive_field_maker.set_graph(graph)

        return self.make_input_for_cnn(node_sequence, receptive_field_maker)

graph =  ig.Read_GraphMLz('./unittest_files/mutag_152.graphml')
node_sequence_size = 18
receptive_field_size = 10
num_channels = 1

preprocessor = DataPreprocessor(node_sequence_size, receptive_field_size, num_channels)
x = preprocessor.execute(graph)
self.assertTrue((1, receptive_field_size, node_sequence_size) == x.shape)