(ns sk-dl4j.binary-ops-datagen)

(defn round-off [^double n]
  (Math/round n))

(defn to-bool
  "Converts a double to boolean. 0 <= _ < 0.5 is false. true otherwise."
  [^double f]
  (let [n (round-off f)]
    (if (= 0 n)
      false
      true)))

(defn rand-noisy-bool
  "Convenience function. Named to denote purpose."
  []
  (rand))

(defn xor [b1 b2]
  (if b1
    (if b2
      false
      true)
    (if b2
      true
      false)))

(defmacro binary-op-sample [op]
  `(let [x1# (rand-noisy-bool)
         x2# (rand-noisy-bool)
         op-val# (~op (to-bool x1#) (to-bool x2#))
         op-val-as-int# (if op-val# 1 0)]
     [x1# x2# op-val-as-int#]))

(defn an-xor-sample []
  (binary-op-sample xor))

(defn an-and-sample []
  (binary-op-sample and))

(defn an-or-sample []
  (binary-op-sample or))

(defn xor-sample-stream []
  (repeatedly an-xor-sample))

(defn and-sample-stream []
  (repeatedly an-and-sample))

(defn or-sample-stream []
  (repeatedly an-or-sample))