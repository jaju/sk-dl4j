;; Helper namespace to generate sample data. For one-off uses.
(ns sk-dl4j.binary-ops-datagen
  (:require [clojure.data.csv :as csv]))

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

(defn xor-sample-stream-fn []
  (repeatedly an-xor-sample))

(defn and-sample-stream-fn []
  (repeatedly an-and-sample))

(defn or-sample-stream-fn []
  (repeatedly an-or-sample))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defmacro dump-to-csv-file
  "Given a stream function, returning N-tuples, `take` suggested number of entries and save them as CSV to the
  supplied file-path."
  [stream-fn outfile num-data]
  `(with-open [out# (clojure.java.io/writer ~outfile)]
     (csv/write-csv out# (take ~num-data (~stream-fn)))))

(defn write-xor-sample [outfile num-data]
  (dump-to-csv-file xor-sample-stream-fn outfile num-data))

(defn write-and-sample [outfile num-data]
  (dump-to-csv-file and-sample-stream-fn outfile num-data))

(defn write-or-sample [outfile num-data]
  (dump-to-csv-file or-sample-stream-fn outfile num-data))
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
