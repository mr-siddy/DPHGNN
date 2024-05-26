import dhg
from dhg.data import CoauthorshipCora, CoauthorshipDBLP, CocitationCora, CocitationCiteseer

class BenchmarkDatasets():
    def __init__(self, name):
        super(BenchmarkDatasets, self).__init__()
        self.dataset = name
        i = '()'
        setattr(self.dataset, f'dataset{i}', name)
    
    def load_hypergraph(self):
        data = getattr(self, self.dataset)
        return data
