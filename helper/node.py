'''
Created on 22.02.2018

@author: ludewig
'''

class PatternNode: 
    
    def __init__(self, id):
        self.children = {}
        self.key = id
        self.count = 0
        self.depth = 0
        self.width = {}
    
    def add(self, seq):
        if len(seq) > 0:
            item = seq[0]
            if not item in self.children:
                self.children[item] = PatternNode(item)
            if len( seq ) == 1:
                self.children[item].count += 1
            self.children[item].add( seq[1:] )
        
    def prune(self, min_count=2, node=None, level=0, first=None):
        
        if node is None:
            nlevel = level + 1
            self.depth = 0
            self.width = {}
            for child in self.children.items():
                child = child[1]
                child.depth = 0
                self.prune( min_count, child, level+1, first=child )
            self.width[nlevel] = len( self.children )
            
            ldel = []
            
            for child in self.children.items():
                child = child[1]
                if child.depth <= 1:
                    ldel.append( child.key )
            
            for d in ldel:
                del self.children[d]
        else:
            nlevel = level + 1
            if nlevel not in self.width :
                self.width[nlevel] = 0
            self.width[nlevel] += len( node.children )
            
            if level > self.depth:
                self.depth = level
                
            if level > first.depth:
                first.depth = level
            
            if level >= 1:
                ldel = []
                for child in node.children.items():
                    child = child[1]
                    if child.count >= min_count:
                        self.prune( min_count, child, level+1, first )
                    else:
                        ldel.append( child.key )
                for d in ldel:
                    del node.children[d]
            else:
                for child in node.children.items():
                    child = child[1]
                    self.prune( min_count, child, level+1, first )
        
            
    def __str__(self, level=0):
        ret = "\t"*level + repr(self.key) + '('+ str(self.count) +')' + "\n"
        for child in self.children.items():
            ret += child[1].__str__(level+1)
        return ret

        