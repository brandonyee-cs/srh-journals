## 9/3/25

I designed what is effectively a high-level overview of the eventual modules in my codebase as a general pipeline (ie how different proposed parts of the system will work).

Ex: raw social media data goes to the multimodal encoder and its subparts (textual encoder, temporal & visual encoder, graph encoder) to be transformed into a usable form before being passed to the cascade attention module.

## 9/4/25

I literally did exactly what it sounds like: I opened up datasets and read through them to ensure completeness and begin to understand their format for later usage.

## 9/5/25

Similar to the overall high-level architecture, I designed the same for the data flow as a subarchitecture, both as a technical overview and as a literal, more simplified version.

## 9/6/25

Based on the dataset and read papers, I devised what I would need to do in various processes to get the required dataset, and then the actions that I would need to take on said dataset to get ideal results. This included not showing tensor calculations for model dimensions (the dimensions of the unified vector in which the matrix passed to the model fit the model).

## 9/8/25

On this day, I first generalized the attention mechanism in a visualization to help you understand. (Closer to 1 is preferred by the model for accuracy.)

I then proposed and proved the coordination score formula. This draws its origins from behavioral finance and behavioral informatics, in which I curmized the extent to which groups were coordinating to post, which helps diagnose whether or not a post is "inauthentic." Extensive proofing is necessary for publication.

9/9/25

Designed validation methods which determine model biases etc. The two chosen were Ground Truth Hierarchy and Cross Validation; these have utility in both optimization and validation, and are industry standards.

I also create ideal success metrics and some quick implementation notes.
