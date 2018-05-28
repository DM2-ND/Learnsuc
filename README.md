# learnsuc
Multi-Type Itemset Embedding for Learning Behavior Success

Accepted by ACM SIGKDD 2018

Authors: Daheng Wang, Meng Jiang, Qingkai Zeng, Zachary Eberhart, Nitesh V. Chawla

Emails: {dwang8, mjiang2, qzeng, zeberhar, nchawla}@nd.edu

Abstract: Contextual behavior modeling use data from multiple contexts to discover patterns for predictive analysis. However, existing behavior prediction models often face difficulties when scaling for massive datasets or handling the sparsity of multicontextual data. In this work, we formulate a behavior as a set of context items of different types (such as operators, goals, and resources), consider an observable itemset as a behavior success, and propose a novel scalable method, "multi-type itemset embedding", to learn the context items' representations preserving the success structure. Unlike most of the existing embedding methods that learn pair-wise proximity from connections between a behavior and one of its items, our method learns item embeddings collectively from interaction among all multi-type items of a behavior, based on which we develop a novel framework, LEARNSUC, for (1) predicting the success rate of any set of items and (2) finding complementary items which maximize the probability of success when incorporated into an itemset. Reproducible experiments demonstrate both effectiveness and efficiency of the proposed framework on these two tasks.
