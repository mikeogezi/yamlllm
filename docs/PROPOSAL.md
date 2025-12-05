# CS 842 Programming Project Proposal

Michael Ogezi (#20926982)

## A high-level declarative DSL for transformer architectures

### **Overview and Objectives**

Transformer-based deep learning models have proven very useful for a variety of applications, like ChatGPT and AlphaFold. However, defining these model architectures often requires writing verbose code in frameworks, like PyTorch. I aim to improve this workflow by creating a high-level, declarative Domain-Specific Language (DSL) that allows users to specify a complete Transformer architecture concisely. The objective is to separate the what (the model's structure) from the how (the underlying framework implementation). 

### **Approach**

I will develop a declarative DSL using YAML syntax for specifying Transformer architectures. This will allow a user to define components, like encoder/decoder stacks, attention mechanisms, position encoding, and feed-forward networks at a high level. A compiler will then transpile this YAML definition into a human-readable, optimized Python script that implements the specified model using PyTorch or JAX. This generated file can then be imported and used in training pipelines. 

### **Milestones**

- Finalize the YAML schema for the DSL and implement a parser to load and validate model definitions. 
- Develop the transpiler backend to generate PyTorch/JAX code for NN layers and activations. 
- Implement the code generation for the complex, domain-specific modules of the Transformer, primarily the multi-head attention and position-wise feed-forward blocks. 
- Write the logic to assemble the generated modules into a complete model class, such as nn.Module in PyTorch. 
- Verify the compiler by transpiling a complete model definition and ensuring the generated PyTorch code can be used. 
- Prove the correctness of the entire pipeline by successfully training a generated model on a simple dataset. 
- Document the DSL syntax and compiler usage and package the entire project with examples for final submission. 