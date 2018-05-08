Network representation in Teneto
--------------------------------

There are two ways that network's are represented in Teneto:

1. As a graphlet/snapshot
2. Contact representation

This tutorial goes through what these different representations are and how to translate between them.

Converting between representations
==================================

Converting between the two different network representations is quite easy.

.. code-block:: python

  import teneto
  G = 1
  C = teneto.utils.contact2graphlet(G)

.. code-block:: python

  G = teneto.utils.graphlet2contact(C)
