(ns sk-dl4j.core
  (:require [clojure.data.csv :as csv]
            [sk-dl4j.binary-ops-datagen :as binary-ops-datagen]))

(defmacro write-csv-file [stream-fn outfile num-data]
  `(with-open [out# (clojure.java.io/writer ~outfile)]
     (csv/write-csv out# (take ~num-data (~stream-fn)))))

(defn write-xor-sample [outfile num-data]
  (write-csv-file binary-ops-datagen/xor-sample-stream outfile num-data))

(defn write-and-sample [outfile num-data]
  (write-csv-file binary-ops-datagen/and-sample-stream outfile num-data))

(defn write-or-sample [outfile num-data]
  (write-csv-file binary-ops-datagen/or-sample-stream outfile num-data))