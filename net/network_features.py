from brian2 import PoissonGroup, Synapses, NeuronGroup, BrianObject


def strong_mem_noise(tr, GExc: NeuronGroup, GInh: NeuronGroup) -> [int]:
    """
        Special kind of Poisson membrane noise that is so strong that it immediately enables
        the post-synaptic neuron.

        Related parameters:

        - :data:`net.standard_params.strong_mem_noise_active`
        - :data:`net.standard_params.strong_mem_noise_rate`
    """

    def strong_mem_noise_on_group(G: NeuronGroup):
        GNoise = PoissonGroup(G.N, rates=tr.strong_mem_noise_rate)
        SynNoise = Synapses(source=GNoise, target=G, on_pre="V_post += 20*mV")
        SynNoise.connect(condition="i==j")
        return [GNoise, SynNoise]

    return strong_mem_noise_on_group(GExc) + strong_mem_noise_on_group(GInh)
