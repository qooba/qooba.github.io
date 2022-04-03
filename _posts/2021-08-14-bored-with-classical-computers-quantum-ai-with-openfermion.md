---
id: 685
title: 'Bored with classical computers?  &#8211; Quantum AI with OpenFermion'
date: '2021-08-14T00:31:20+02:00'
author: qooba
layout: post
guid: 'https://qooba.net/?p=685'
permalink: /2021/08/14/bored-with-classical-computers-quantum-ai-with-openfermion/
categories:
    - 'No classified'
tags:
    - Cirq
    - OpenFermion
    - Python
    - 'Quantum Computing'
    - QuantumAI
---

<img src="{{ site.relative_url }}wp-content/uploads/2021/08/resulta-7-1767629_1280.jpg" alt="calculator" width="900" />

In this article I will show how we can prepare and perform calculations on quantum computers
using [OpenFermion](https://quantumai.google/openfermion), [Cirq](https://quantumai.google/cirq) and [PySCF](https://pyscf.org/).

Before you will continue reading please watch short introduction: 

https://www.youtube.com/watch?v=dR7GgJGafQQ

Currently, there are many supercomputing centers, where we can run complicated simulations.
However, there are still problems that are beyond the capabilities of classical computers,
which can be addressed by quantum computers.

<img src="https://qooba.net/wp-content/uploads/2021/08/QuantumAIOpenFermion.00-scaled.jpeg" alt="materials science" width="900" />

Quantum chemistry and materials science problems which that are described by the laws of 
quantum mechanics can be mapped to the quantum computers and projected to qubits. 

[OpenFermion](https://quantumai.google/openfermion) is the library which can help to perform such calculations on a quantum computer.

Additionally we will use the PySCF package which will help to perform initial structure optimization (if you are interested in PySCF package I have shared the example DFT based band structure calculation of the single layer graphene structure [pyscf_graphene.ipynb](https://github.com/qooba/quantumai-openfermion/blob/main/notebooks/pyscf_graphene.ipynb)).

<img src="https://qooba.net/wp-content/uploads/2021/08/QuantumAIOpenFermion.01-scaled.jpeg" alt="materials science" width="900" />

In our example we will investigate [latex]H_2[/latex] molecule for simplicity. We will use the PySCF package to find optimal bond length of the molecule. 

Thanks to the [OpenFermion-PySCF](https://github.com/quantumlib/OpenFermion-PySCF) plugin we can smoothly use the molecule initial state obtained from PySCF package run in OpenFermion library ([openfermionpyscf_h2.ipynb](https://github.com/qooba/quantumai-openfermion/blob/main/notebooks/openfermionpyscf_h2.ipynb)).

```python
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf

geometry = create_molecule(bond_length)
basis = 'sto-3g'
multiplicity = 1

run_scf = 1
run_mp2 = 1
run_cisd = 0
run_ccsd = 0
run_fci = 1

molecule = MolecularData(geometry, basis, multiplicity)
 
# Run pyscf.
molecule = run_pyscf(molecule,
                     run_scf=run_scf,
                     run_mp2=run_mp2,
                     run_cisd=run_cisd,
                     run_ccsd=run_ccsd,
                     run_fci=run_fci)

```


<img src="https://qooba.net/wp-content/uploads/2021/08/QuantumAIOpenFermion.02-scaled.jpeg" alt="materials science" width="900" />

Now it is time to compile the molecule to the representation readable by the quantum computer
using [OpenFermion](https://quantumai.google/openfermion) and [Cirq](https://quantumai.google/cirq) library.
Currently you can use several methods to achieve this:
* [Constructing a Basis Change Circuits](https://quantumai.google/openfermion/tutorials/circuits_1_basis_change)
* [Constructing a Diagonal Coulomb Trotter Step](https://quantumai.google/openfermion/tutorials/circuits_2_diagonal_coulomb_trotter)
* [Constructing Trotter Steps With Low-Rank Decomposition](https://quantumai.google/openfermion/tutorials/circuits_3_arbitrary_basis_trotter)

Using one of this methods we get optimized quantum circuit.
In our case [the quantum cirquit](https://quantumai.google/cirq/circuits) for [latex]H_2[/latex] system will be represented by **4 qubits** and **operations** that act on them (**moment** is collection of **operations** that act at the same abstract time slice).

<img src="https://qooba.net/wp-content/uploads/2021/08/QuantumAIOpenFermion.03-scaled.jpeg" alt="materials science" width="900" />

Finally we can use quantum circuit to run the calculations on [the cirq simulator](https://quantumai.google/cirq/simulation) or on [the real quantum computer](https://quantumai.google/quantum-computing-service).
