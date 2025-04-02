# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

class Base_Model(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def predict(self, messages):
        pass

    @abstractmethod
    def costs(self, input_tokens, output_tokens):
        pass




