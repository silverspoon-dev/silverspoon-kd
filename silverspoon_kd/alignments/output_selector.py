"""Selector callable for extracting a single element from tuple outputs."""


class OutputSelector:
    """Extracts an element at a given index from tuple outputs.

    Many PyTorch modules (e.g., attention layers) return tuples like
    ``(hidden_states, attention_weights)``. This selector picks one element
    for comparison in an alignment. If the output is not a tuple, it is
    returned as-is.
    """

    def __init__(self, index: int = 0):
        """Initialize the selector.

        Args:
            index: The index of the element to extract from tuple outputs.
                Default: ``0`` (the first element).
        """
        self.index = index

    def __call__(self, output):
        """Extract the element at ``self.index`` from the output.

        Args:
            output: Module output. If a tuple, the element at ``self.index``
                is returned. Otherwise the output is returned unchanged.

        Returns:
            The selected element or the original output.
        """
        if isinstance(output, tuple):
            return output[self.index]
        return output
