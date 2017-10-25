(ns sk-dl4j.xor-nn
  (:import (org.datavec.api.records.reader.impl.csv CSVRecordReader)
           (org.datavec.api.split FileSplit)
           (java.io File)
           (org.nd4j.linalg.dataset.api.iterator DataSetIterator)
           (org.deeplearning4j.datasets.datavec RecordReaderDataSetIterator)
           (org.deeplearning4j.nn.conf NeuralNetConfiguration NeuralNetConfiguration$Builder Updater NeuralNetConfiguration$ListBuilder)
           (org.deeplearning4j.nn.api OptimizationAlgorithm)
           (org.deeplearning4j.nn.conf.layers DenseLayer$Builder OutputLayer$Builder)
           (org.nd4j.linalg.activations Activation)
           (org.deeplearning4j.nn.weights WeightInit)
           (org.deeplearning4j.nn.conf.distribution UniformDistribution)
           (org.nd4j.linalg.lossfunctions LossFunctions$LossFunction)
           (org.deeplearning4j.nn.multilayer MultiLayerNetwork)
           (org.deeplearning4j.optimize.listeners ScoreIterationListener)
           (org.deeplearning4j.eval Evaluation)))

(defonce train-file "train-xor.csv")
(defonce test-file "test-xor.csv")

(defn csv-dataset-iterator [file-path & {:keys [batch-size label-index label-cardinality]}]
  (let [rr         (CSVRecordReader.)
        file-split (FileSplit. (File. (.getFile (clojure.java.io/resource file-path))))
        _          (.initialize rr file-split)
        iterator   (RecordReaderDataSetIterator. rr batch-size label-index label-cardinality)]
    iterator))

(defn build-dense-layer [& {:keys [n-in n-out activation-fn weight-init dist]}]
  (let [b (DenseLayer$Builder.)]
    (doto b
      (.nIn n-in)
      (.nOut n-out)
      (.activation activation-fn)
      (.weightInit weight-init)
      (.dist dist))
    (.build b)))

(defn build-output-layer [& {:keys [loss-fn n-in n-out activation-fn weight-init dist]}]
  (let [b (OutputLayer$Builder. loss-fn)]
    (doto b
      (.nIn n-in)
      (.nOut n-out)
      (.activation activation-fn)
      (.weightInit weight-init)
      (.dist dist))
    (.build b)))

(defn create-multilayer-configuration [& {:keys [seed num-iterations optimization-algorithm
                                                 learning-rate updater use-drop-connect? bias-init
                                                 mini-batch?]
                                          :or   {seed                   1023
                                                 num-iterations         1000
                                                 optimization-algorithm OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT
                                                 learning-rate          0.01
                                                 updater                Updater/NESTEROVS
                                                 use-drop-connect?      false
                                                 bias-init              0
                                                 mini-batch?            false}}]
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

(comment
  (csv-dataset-iterator "train-xor.csv" :batch-size 50 :label-index 2 :label-cardinality 2)
  (csv-dataset-iterator "test-xor.csv" :batch-size 50 :label-index 2 :label-cardinality 2)

  (let [conf-builder                           (create-multilayer-configuration)
        ^NeuralNetConfiguration$ListBuilder lb (.list conf-builder)
        hidden-layer                           (build-dense-layer :n-in 2 :n-out 4 :activation-fn Activation/SIGMOID
                                                                  :weight-init WeightInit/DISTRIBUTION
                                                                  :dist (UniformDistribution. 0 1))
        output-layer                           (build-output-layer :loss-fn LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD
                                                                   :n-in 4
                                                                   :n-out 2
                                                                   :activation-fn Activation/SOFTMAX
                                                                   :weight-init WeightInit/DISTRIBUTION
                                                                   :dist (UniformDistribution. 0 1))]
    (.layer lb 0 hidden-layer)
    (.layer lb 1 output-layer)
    (.pretrain lb false)
    (.backprop lb true)
    (let [conf                   (.build lb)
          ^MultiLayerNetwork net (MultiLayerNetwork. conf)
          _                      (.init net)
          _                      (.setListeners net [(ScoreIterationListener. 100)])
          layers                 (.getLayers net)]

      (println (str "Number of parameters in layer 0 " (.numParams (aget layers 0))))
      (println (str "Number of parameters in layer 1 " (.numParams (aget layers 1))))

      (.fit net (csv-dataset-iterator "train-xor.csv" :batch-size 50 :label-index 2 :label-cardinality 2))

      (let [test-iter (csv-dataset-iterator "test-xor.csv" :batch-size 50 :label-index 2 :label-cardinality 2)
            eval (Evaluation. 2)]
        (while (.hasNext test-iter)
          (let [data-set (.next test-iter)
                labels (.getLabels data-set)
                features (.getFeatureMatrix data-set)
                predicted (.output net features false)]
            (.eval eval labels predicted)))
        (println (.stats eval)))
      )))


