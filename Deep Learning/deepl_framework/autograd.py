import numpy as np
from graphviz import Digraph


class Tensor:
    all_tensors = []

    def __init__(self, data, inputs=()):
        """ data: value of this tensor, inputs: list of input tensors """
        self.id = len(self.all_tensors)
        self.data = data
        self.inputs = inputs
        self.grad = np.zeros_like(data)
        self.all_tensors.append(self)

        self.forward_usages = 0
        self.backward_usages = 0
        for input_ in self.inputs:        # It increments to the forward usage of inputs to this tensor
            input_.forward_usages += 1

    def prev(self) -> list:
        return self.inputs

    def print(self):
        print('%d)' % self.id, self.__class__.__name__)
        print('data\n', self.data)
        print('grad\n', self.grad)
        print('inputs', [p.id for p in self.prev()])

    def graph(self, graph=None):
        """ returns a graphviz graph of all dependencies """
        if graph is None:
            graph = Digraph(format='pdf', node_attr=dict(style='filled', shape='rect'))
        graph.node(str(self.id), label=self.name(), fillcolor='white' if len(self.inputs) == 0 else 'lightblue')
        for ins in self.inputs:
            ins.graph(graph)
            graph.edge(str(ins.id), str(self.id))
        return graph

    def name(self) -> str:
        return '%d: %s' % (self.id, self.__class__.__name__)
        
    @classmethod
    def reset_tensors(cls):
        cls.all_tensors = []

    @classmethod
    def reset_grads(cls):
        for tensor in cls.all_tensors:
            tensor.grad.fill(0)

    def backward(self, start=True):
        """ Recursively back-propagate gradients """
        # TODO (a)
        # if we start backpropagating here, each value has a gradient of one
        # call the _assign_grads() method once all Tensors that use this one as input have assigned their gradients
        #   hint: self.forward_usages, self.backward_usages to track that
        #   then also recursively continue with every input of this tensor
        if(self.forward_usages == self.backward_usages):
            if start == True:
                self.grad = np.ones(self.data.shape)
            self._assign_grads()
            for input_ in self.inputs:
                input_.backward(start=False)
        else:
            pass

    def _assign_grads(self):
        """ Compute and assign gradients for each input """
        raise NotImplementedError


class Variable(Tensor):
    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = name

    def _assign_grads(self):
        pass

    def name(self) -> str:
        return '%d: %s (%s)' % (self.id, self._name, self.__class__.__name__)


class Neg(Tensor):
    def __init__(self, a: Tensor):
        result = -a.data
        super().__init__(result, [a])
        self.a = a

    def _assign_grads(self):
        # add gradients to inputs of this graph node
        self.a.grad -= self.grad
        self.a.backward_usages += 1 


class Sqrt(Tensor):
    def __init__(self, a: Tensor):
        result = np.sqrt(a.data)
        super().__init__(result, [a])
        self.a = a

    def _assign_grads(self):
        # add gradients to inputs of this graph node
        self.a.grad += self.grad / (2 * np.sqrt(self.a.data))
        self.a.backward_usages += 1


class ReduceMean(Tensor):
    def __init__(self, a: Tensor):
        result = np.mean(a.data)
        super().__init__(result, [a])
        self.a = a

    def _assign_grads(self):
        self.a.grad += self.grad/len(self.a.data)
        self.a.backward_usages += 1


class Add(Tensor):
    def __init__(self, *tensors: [Tensor]):
        result = sum([v.data for v in tensors])
        super().__init__(result, tensors)

    def _assign_grads(self):
        for v in self.inputs:
            v.grad += self.grad
            v.backward_usages += 1


class Mul(Tensor):
    def __init__(self, a: Tensor, b: Tensor):
        result = a.data * b.data
        super().__init__(result, [a, b])
        self.a = a
        self.b = b

    def _assign_grads(self):
        self.a.grad += self.grad * self.b.data
        self.b.grad += self.grad * self.a.data
        self.a.backward_usages += 1
        self.b.backward_usages += 1


class MatMul(Tensor):
    def __init__(self, a: Tensor, b: Tensor):
        result = np.matmul(a.data, b.data)
        super().__init__(result, [a, b])
        self.a = a
        self.b = b

    def _assign_grads(self):
        self.a.grad += np.matmul(self.grad, self.b.data.T) 
        self.b.grad += np.matmul(self.a.data.T, self.grad)
        self.a.backward_usages += 1
        self.b.backward_usages += 1

class ReLU(Tensor):
    def __init__(self, a: Tensor):
        result = (a.data > 0) * a.data
        super().__init__(result, [a])
        self.a = a

    def _assign_grads(self):
        temp_a = (self.a.data > 0)
        self.a.grad += self.grad*temp_a
        self.a.backward_usages += 1
        


class Sigmoid(Tensor):
    def __init__(self, a: Tensor):
        result = 1 / (1 + np.exp(-a.data))
        super().__init__(result, [a])
        self.a = a

    def _assign_grads(self):
        self.a.grad += self.grad*self.data*(1-self.data)
        self.a.backward_usages += 1


def mse(a: Tensor, b: Tensor) -> Tensor:
    return ReduceMean(Mul(Add(a, Neg(b)), Add(a, Neg(b))))


def main():
    # fixed input
    v0 = Variable('A', np.array([[-5.0, -2.0, -1.0], [-5.0, -1.0, -3.0]]))

    # example computation graph, built on the fly
    vx = Neg(v0)
    vx = Sqrt(vx)
    
    # start backpropagating
    vx.backward()

    #print gradients
    for v_ in Tensor.all_tensors:
        # if not isinstance(v_, Variable):
        #     continue
        print('-'*50)
        v_.print()
        #print(v_.inputs)

    # show the computation graph
    vx.graph().render('/tmp/autograd/graph', view=True)

    # # reset all tensors, since the graph is only built for a single forward+backward pass
    Tensor.reset_tensors()


if __name__ == '__main__':
    main()
