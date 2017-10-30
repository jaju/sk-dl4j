(ns sk-dl4j.xor-nn
  (:import [java.io File]
           [org.nd4j.linalg.activations Activation]
           [org.nd4j.linalg.dataset.api.iterator DataSetIterator]
           [org.nd4j.linalg.lossfunctions LossFunctions$LossFunction]
           [org.datavec.api.split FileSplit]
           [org.datavec.api.records.reader.impl.csv CSVRecordReader]
           [org.deeplearning4j.datasets.datavec RecordReaderDataSetIterator]
           [org.deeplearning4j.nn.conf NeuralNetConfiguration NeuralNetConfiguration$Builder Updater NeuralNetConfiguration$ListBuilder]
           [org.deeplearning4j.nn.conf.layers DenseLayer$Builder OutputLayer$Builder]
           [org.deeplearning4j.nn.conf.distribution UniformDistribution]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.deeplearning4j.nn.api OptimizationAlgorithm]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.eval Evaluation]
           [org.deeplearning4j.optimize.api IterationListener]))

;; Pre-created training and test data. See the binary-ops-datagen namespace.
(def xor-training-data-file "resources/train-xor.csv")
(def xor-test-data-file "resources/test-xor.csv")

(defn create-csv-dataset-iterator
  "Create an iterator over a CSV-formatted data-set in a file, for use in network train/test operations.
  The input contains the features as well as a label-column. You need to indicate which column
  corresponds to the label, and how many distinct labels exist - the cardinality."
  [file-path & {:keys [batch-size label-index label-cardinality]}]
  (let [rr         (CSVRecordReader.)
        file-split (-> file-path
                       (File.)
                       (FileSplit.))
        _          (.initialize rr file-split)
        iterator   (RecordReaderDataSetIterator. rr batch-size label-index label-cardinality)]
    iterator))

(defn build-dense-layer
  "Just a DenseLayer building helper."
  [& {:keys [n-in n-out activation-fn weight-init dist]}]
  (let [b (DenseLayer$Builder.)]
    (doto b
      (.nIn n-in)
      (.nOut n-out)
      (.activation activation-fn)
      (.weightInit weight-init)
      (.dist dist))
    (.build b)))

(defn build-output-layer
  "OutputLayer building helper."
  [& {:keys [loss-fn n-in n-out activation-fn weight-init dist]}]
  (let [b (OutputLayer$Builder. loss-fn)]
    (doto b
      (.nIn n-in)
      (.nOut n-out)
      (.activation activation-fn)
      (.weightInit weight-init)
      (.dist dist))
    (.build b)))

(defn create-neural-net-configuration-builder
  "Initialize a base NeuralNetConfiguration builder with required hyper-parameters.
  You'd then be adding layers on the returned object before using it to create the network."
  [& {:keys [seed
             num-iterations
             optimization-algorithm
             learning-rate
             updater
             use-drop-connect?
             bias-init
             mini-batch?]}]
  (let [builder (NeuralNetConfiguration$Builder.)]
    (doto builder
      (.seed ^long seed)
      (.iterations num-iterations)
      (.optimizationAlgo optimization-algorithm)
      (.learningRate learning-rate)
      (.updater updater)
      (.useDropConnect use-drop-connect?)
      (.biasInit bias-init)
      (.miniBatch mini-batch?))
    builder))

(defn create-score-evolution-listener
  "Create a simple listener which prints the current score after every iter-count iterations of the network during the
  training phase. You could use any other, but the default dependence on a logging framework of the ScoreIterationListener
  as well as quite a bit of (noisy) output may not be something you like. Mostly a blind mimic of the ScoreIterationListener.
  Also demonstrates how straightforward it is to `do` Java here."
  [iter-count]
  (let [count    (atom 0)
        invoked? (atom false)]
    (reify IterationListener
      (invoke [_] (reset! invoked? true))
      (invoked [_] @invoked?)
      (iterationDone [_ model iteration]
        (swap! count inc)
        (when (zero? (mod @count iter-count))
          (println (str "The score currently is: " (.score model))))))))

(defn- build-neural-net-config [config-builder hidden-layer output-layer]
  (let [list-builder (.list config-builder)]
    (doto list-builder
      (.layer 0 hidden-layer)
      (.layer 1 output-layer)
      (.pretrain false)
      (.backprop true))
    (.build list-builder)))

(defn run
  "Trains, tests, and then returns the model that you can play with further."
  [& {:keys [hl-activation-fn ol-activation-fn loss-fn optimization-algorithm num-iterations]
      :or   {hl-activation-fn Activation/SIGMOID
             ol-activation-fn Activation/SOFTMAX
             loss-fn          LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD
             optimization-algorithm OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT
             num-iterations   500}}]
  (let [config-builder    (create-neural-net-configuration-builder :seed 1023
                                                                   :num-iterations num-iterations
                                                                   :optimization-algorithm optimization-algorithm
                                                                   :learning-rate 0.01
                                                                   :updater Updater/NESTEROVS
                                                                   :use-drop-connect? true
                                                                   :bias-init 0
                                                                   :mini-batch? true)
        hidden-layer      (build-dense-layer :n-in 2
                                             :n-out 4
                                             :activation-fn hl-activation-fn
                                             :weight-init WeightInit/DISTRIBUTION
                                             :dist (UniformDistribution. 0 1))
        output-layer      (build-output-layer :loss-fn loss-fn
                                              :n-in 4
                                              :n-out 2
                                              :activation-fn ol-activation-fn
                                              :weight-init WeightInit/DISTRIBUTION
                                              :dist (UniformDistribution. 0 1))
        neural-net-config (build-neural-net-config config-builder hidden-layer output-layer)
        net               (doto (MultiLayerNetwork. neural-net-config)
                            (.init)
                            (.setListeners [(create-score-evolution-listener 100)]))
        layers            (.getLayers net)]

    (println (str "Number of parameters in layer 0 " (.numParams (aget layers 0))))
    (println (str "Number of parameters in layer 1 " (.numParams (aget layers 1))))

    (.fit net (create-csv-dataset-iterator xor-training-data-file :batch-size 50 :label-index 2 :label-cardinality 2))

    (let [test-iter (create-csv-dataset-iterator xor-test-data-file :batch-size 50 :label-index 2 :label-cardinality 2)
          eval      (Evaluation. 2)]
      (while (.hasNext test-iter)
        (let [data-set  (.next test-iter)
              labels    (.getLabels data-set)
              features  (.getFeatureMatrix data-set)
              predicted (.output net features false)]
          (.eval eval labels predicted)))
      (println (.stats eval)))
    net))
