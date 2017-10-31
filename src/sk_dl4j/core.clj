(ns sk-dl4j.core
  (:require [sk-dl4j.xor-nn :as xor-nn])
  (:import (org.nd4j.linalg.activations Activation)))

(comment
  ;; Run me! REPL-up, and evaluate the following form. Nothing else to be configured.
  ;; It returns the trained model too. You can capture that in a var to play with in the REPL.
  (xor-nn/run :hl-activation-fn Activation/HARDTANH
              :ol-activation-fn Activation/SOFTMAX))