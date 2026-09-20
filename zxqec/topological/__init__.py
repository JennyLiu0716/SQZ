# zxqec - space-time ZX-diagrams for dynamic quantum error correction
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pauli-box hypergraph normal form and the topological view.

A stabilizer of weight *w* is represented as one off-qubit phase-free X spider
connected to *w* on-qubit phase-free Z spiders (the Pauli boxes of Def. 5.1 of
arXiv:2603.09580). A diagram in this normal form is a hypergraph whose
hyperedges are labelled by Pauli strings. Planned contents:

* rewriting a stabilizer diagram into the normal form and identifying the
  stabilizers;
* the topological view: each hyperedge drawn as a surface (vertices in a
  plane) or a volume (vertices in 3D), coloured red for X, green for Z, blue
  for Y, with gradients for mixed Pauli strings; toggling between the ZX view
  and the topological view.

"""
