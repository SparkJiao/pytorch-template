# pytorch-template
### Intro
This is a simple template for research, aiming at fast coding. To use this template, you need to:
- Write your own data set reader. The only request for it is that the reader itself need to implement `__iter__` method, so that
the trainer can iterate on it directly.
- Write your own model. The model should implement the method in abstract class `Controller`. If you have several models
with the same logic in logging and generate inputs, i.e. models for the same dataset, you can wrap another high level
controller on them.
- For other requests, you could implement another trainer based on the simplest `Trainer`. For fast iteration on different experiments,
you could extract some abstract method and implement different functions as extra parameters, i.e. if you need different optimization
strategies for different model, you could implement a high level method to provide a optimizer and learning rate scheduler, with different
methods defining parameter groups.
