(ns sk-dl4j.core
  (:require [sk-dl4j.xor-nn :as xor-nn])
  (:import (org.nd4j.linalg.activations Activation)))

(comment
  (xor-nn/run :hl-activation-fn Activation/HARDTANH
              :ol-activation-fn Activation/SOFTMAX))