(defproject sk-dl4j "0.1.0-SNAPSHOT"
  :description "Starter Kit: deeplearning4j"
  :url "https://starter-kit.msync.org/deeplearning4j"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0-beta3"]
                 [org.clojure/data.csv "0.1.4"]
                 [org.deeplearning4j/deeplearning4j-core "0.9.1"]
                 [org.nd4j/nd4j-native-platform "0.9.1"]
                 [org.datavec/datavec-api "0.9.1"]
                 [midje "1.9.0-alpha9" :exclusions [org.clojure/clojure]]]

  :java-source-paths ["java"]

  :jvm-opts ["-Xmx4g"])
