Introduction
============

This package provides a framework for modeling seasonal vaccine uptake, such as influenza vaccines. It has several key features to support modeling vaccine uptake patterns and addressing questions that public health practitioners may have, including:

- Ability to incorporate arbitrary categorical covariates, including geographic regions.
- Support for observations at multiple time scales, such as daily, weekly, or monthly as well as irregular observations.
- Capability to model the effects of interventions, such as changes in vaccine policy or public health campaigns.
- Flexibility to model different types of vaccine uptake patterns through the use of curve families.

The package does require some familiarity with Python to use effectively, but it is designed to be user-friendly and accessible to public health practitioners. It provides a set of tools and functions that can be used to create models, analyze data, and visualize results.

In particular this package is designed to help users address questions such as:

- How does vaccine uptake vary across different demographic groups or geographic regions?
- What is the impact of interventions on vaccine uptake patterns?
- How do seasonal patterns in vaccine uptake change over time?
- What are reasonable ranges to expect for vaccine uptake for a season?

The package is built on top of `pymc`, providing an interface that allows users to define models using a high-level API that are translated to `pymc` models under the hood. This allows users to take advantage of the powerful probabilistic programming capabilities of `pymc` while still being able to work with a user-friendly interface.
