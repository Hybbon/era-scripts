package edu.ucsc.cs.model;

import java.text.DecimalFormat;

import edu.umd.cs.psl.application.inference.LazyMPEInference
import edu.umd.cs.psl.application.inference.MPEInference;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxPseudoLikelihood;
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin;
import edu.umd.cs.psl.config.*
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.database.DatabasePopulator;
import edu.umd.cs.psl.database.Partition;
import edu.umd.cs.psl.database.ReadOnlyDatabase;
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.memory.MemoryFullInferenceResult
import edu.umd.cs.psl.groovy.PSLModel;
import edu.umd.cs.psl.groovy.PredicateConstraint;
import edu.umd.cs.psl.groovy.SetComparison;
import edu.umd.cs.psl.model.argument.ArgumentType;
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.argument.UniqueID;
import edu.umd.cs.psl.model.argument.Variable;
import edu.umd.cs.psl.model.atom.GroundAtom;
import edu.umd.cs.psl.model.function.ExternalFunction;
import edu.umd.cs.psl.ui.functions.textsimilarity.*
import edu.umd.cs.psl.ui.loading.InserterUtils;
import edu.umd.cs.psl.util.database.Queries;


ConfigManager cm = ConfigManager.getManager()
ConfigBundle config = cm.getBundle("basic-example")

def defaultPath = System.getProperty("java.io.tmpdir")
String dbpath = config.getString("dbpath", defaultPath + File.separator + "rec_sys_100k")
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), config)

sq = true;	//when sq=false then the potentials are linear while when sq is true the potentials are squared

Double totalRMSE = 0.0;
Double totalMAE = 0.0;

int num_folds = 5;

for(int fold=1;fold<=num_folds;fold++){

    PSLModel m = new PSLModel(this, data)

        //DEFINITION OF THE MODEL
        //general predicates
        m.add predicate: "user", 			types: [ArgumentType.UniqueID]
        m.add predicate: "item",			types: [ArgumentType.UniqueID]
        m.add predicate: "rating",			types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
        m.add predicate: "rated",			types: [ArgumentType.UniqueID, ArgumentType.UniqueID] //this is used in the blocking mechanism

        m.add predicate: "bprmf_rating", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
        m.add rule: ((bprmf_rating(U,I)) >> rating(U,I)), weight:3, squared:sq;
        m.add rule: ((rating(U,I)) >> bprmf_rating(U,I)), weight:3, squared:sq;

        m.add predicate: "bprslim_rating", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
        m.add rule: ((bprslim_rating(U,I)) >> rating(U,I)), weight:3, squared:sq;
        m.add rule: ((rating(U,I)) >> bprslim_rating(U,I)), weight:3, squared:sq;

        m.add predicate: "cofirank_rating", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
        m.add rule: ((cofirank_rating(U,I)) >> rating(U,I)), weight:3, squared:sq;
        m.add rule: ((rating(U,I)) >> cofirank_rating(U,I)), weight:3, squared:sq;

        m.add predicate: "itemknn_rating", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
        m.add rule: ((itemknn_rating(U,I)) >> rating(U,I)), weight:3, squared:sq;
        m.add rule: ((rating(U,I)) >> itemknn_rating(U,I)), weight:3, squared:sq;

        m.add predicate: "leastsquareslim_rating", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
        m.add rule: ((leastsquareslim_rating(U,I)) >> rating(U,I)), weight:3, squared:sq;
        m.add rule: ((rating(U,I)) >> leastsquareslim_rating(U,I)), weight:3, squared:sq;

        m.add predicate: "mostpopular_rating", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
        m.add rule: ((mostpopular_rating(U,I)) >> rating(U,I)), weight:3, squared:sq;
        m.add rule: ((rating(U,I)) >> mostpopular_rating(U,I)), weight:3, squared:sq;

        m.add predicate: "multicorebprmf_rating", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
        m.add rule: ((multicorebprmf_rating(U,I)) >> rating(U,I)), weight:3, squared:sq;
        m.add rule: ((rating(U,I)) >> multicorebprmf_rating(U,I)), weight:3, squared:sq;

        m.add predicate: "softmarginrankingmf_rating", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
        m.add rule: ((softmarginrankingmf_rating(U,I)) >> rating(U,I)), weight:3, squared:sq;
        m.add rule: ((rating(U,I)) >> softmarginrankingmf_rating(U,I)), weight:3, squared:sq;

        m.add predicate: "puresvd_rating", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
        m.add rule: ((puresvd_rating(U,I)) >> rating(U,I)), weight:3, squared:sq;
        m.add rule: ((rating(U,I)) >> puresvd_rating(U,I)), weight:3, squared:sq;

        m.add predicate: "wrmf_rating", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
        m.add rule: ((wrmf_rating(U,I)) >> rating(U,I)), weight:3, squared:sq;
        m.add rule: ((rating(U,I)) >> wrmf_rating(U,I)), weight:3, squared:sq;

        println m;



        //keep track of the time
        TimeNow = new Date();
        println "start time is " + TimeNow
            int num_similar_items = 50;
        int num_similar_users = 50;
        int ext = 60;
        def dir = 'data'+java.io.File.separator+'hyper_out'+java.io.File.separator;
        def wdir = dir + "weight_learning" + java.io.File.separator // + 'weights' + java.io.File.separator;

        //we put in the same partition things that are observed
        def evidencePartition = new Partition(0 + fold * num_folds);	     // observed data for weight learning
        def targetPartition = new Partition(1 + fold * num_folds);		     // unobserved data for weight learning
        Partition trueDataPartition = new Partition(2 + fold * num_folds);  // train set for inference
        def evidencePartition2 = new Partition(3 + fold * num_folds);		 // test set for inference
        def targetPartition2 = new Partition(4 + fold * num_folds);

        def insert = data.getInserter(user, evidencePartition)
            InserterUtils.loadDelimitedData(insert, dir + "users");

        insert = data.getInserter(item, evidencePartition)
            InserterUtils.loadDelimitedData(insert, dir + "items");

        insert = data.getInserter(rated, evidencePartition)
            InserterUtils.loadDelimitedData(insert, wdir + "u" + fold + ".rated");

        insert = data.getInserter(rating, evidencePartition);
        InserterUtils.loadDelimitedDataTruth(insert, wdir + "u" + fold + ".train");

        // BASELINES

        insert = data.getInserter(bprmf_rating, evidencePartition)
            InserterUtils.loadDelimitedDataTruth(insert, wdir + "u" + fold + "-BPRMF.out");

        insert = data.getInserter(bprslim_rating, evidencePartition)
            InserterUtils.loadDelimitedDataTruth(insert, wdir + "u" + fold + "-BPRSLIM.out");

        insert = data.getInserter(cofirank_rating, evidencePartition)
            InserterUtils.loadDelimitedDataTruth(insert, wdir + "u" + fold + "-CofiRank.out");

        insert = data.getInserter(itemknn_rating, evidencePartition)
            InserterUtils.loadDelimitedDataTruth(insert, wdir + "u" + fold + "-ItemKNN.out");

        insert = data.getInserter(leastsquareslim_rating, evidencePartition)
            InserterUtils.loadDelimitedDataTruth(insert, wdir + "u" + fold + "-LeastSquareSLIM.out");

        insert = data.getInserter(mostpopular_rating, evidencePartition)
            InserterUtils.loadDelimitedDataTruth(insert, wdir + "u" + fold + "-MostPopular.out");

        insert = data.getInserter(multicorebprmf_rating, evidencePartition)
            InserterUtils.loadDelimitedDataTruth(insert, wdir + "u" + fold + "-MultiCoreBPRMF.out");

        insert = data.getInserter(softmarginrankingmf_rating, evidencePartition)
            InserterUtils.loadDelimitedDataTruth(insert, wdir + "u" + fold + "-SoftMarginRankingMF.out");

        insert = data.getInserter(puresvd_rating, evidencePartition)
            InserterUtils.loadDelimitedDataTruth(insert, wdir + "u" + fold + "-pureSVD.out");

        insert = data.getInserter(wrmf_rating, evidencePartition)
            InserterUtils.loadDelimitedDataTruth(insert, wdir + "u" + fold + "-WRMF.out");

        // to predict
        insert = data.getInserter(rating, targetPartition)
            InserterUtils.loadDelimitedData(insert, wdir + "u" + fold + ".topredict");

        println "Inserted all data for weight learning"

        //target partition
        Database db = data.getDatabase(targetPartition, [user, item, rated,
                bprmf_rating,
                bprslim_rating,
                cofirank_rating,
                itemknn_rating,
                leastsquareslim_rating,
                mostpopular_rating,
                multicorebprmf_rating,
                softmarginrankingmf_rating,
                puresvd_rating,
                wrmf_rating
        ] as Set, evidencePartition);

        println "Successfully created target partition"

        //learn the weights
        //-----------------start
        /* learn the weights from data. For that, we need to have some
         * evidence data from which we can learn. In our example, that means we need to
         * specify ratings, which we now load into another partition.
         */
        insert = data.getInserter(rating, trueDataPartition)
            InserterUtils.loadDelimitedDataTruth(insert, dir + "u" + fold + ".test");
        /* Now, we can learn the weights.
         * We first open a database which contains all the target atoms as observations.
         * We then combine this database with the original database to learn.
         */

        println "Start weight learning..."
            Database trueDataDB = data.getDatabase(trueDataPartition, [rating] as Set);

        MaxLikelihoodMPE weightLearning = new MaxLikelihoodMPE(m, db, trueDataDB, config);

        weightLearning.learn();
        weightLearning.close();

        //print the new model
        println ""
            println "Learned model:"
            println m
            db.close();	//close this db as we will not use it again

        //keep track of the time
        TimeNow = new Date();
        println "time after weight learning is " + TimeNow


            //perform inference with weight learning
            //we put in the same partition things that are observed

        println "users"
        insert = data.getInserter(user, evidencePartition2)
            InserterUtils.loadDelimitedData(insert, dir + "users");

        println "items"
        insert = data.getInserter(item, evidencePartition2)
            InserterUtils.loadDelimitedData(insert, dir + "items");

        insert = data.getInserter(rated, evidencePartition2)
            InserterUtils.loadDelimitedData(insert, dir + "rated");

        insert = data.getInserter(rating, evidencePartition2);
            InserterUtils.loadDelimitedDataTruth(insert, dir + "u" + fold + ".train");

        // BASELINES
        insert = data.getInserter(bprmf_rating, evidencePartition2)
            InserterUtils.loadDelimitedDataTruth(insert, dir + "u" + fold + "-BPRMF.out");

        insert = data.getInserter(bprslim_rating, evidencePartition2)
            InserterUtils.loadDelimitedDataTruth(insert, dir + "u" + fold + "-BPRSLIM.out");

        insert = data.getInserter(cofirank_rating, evidencePartition2)
            InserterUtils.loadDelimitedDataTruth(insert, dir + "u" + fold + "-CofiRank.out");

        insert = data.getInserter(itemknn_rating, evidencePartition2)
            InserterUtils.loadDelimitedDataTruth(insert, dir + "u" + fold + "-ItemKNN.out");

        insert = data.getInserter(leastsquareslim_rating, evidencePartition2)
            InserterUtils.loadDelimitedDataTruth(insert, dir + "u" + fold + "-LeastSquareSLIM.out");

        insert = data.getInserter(mostpopular_rating, evidencePartition2)
            InserterUtils.loadDelimitedDataTruth(insert, dir + "u" + fold + "-MostPopular.out");

        insert = data.getInserter(multicorebprmf_rating, evidencePartition2)
            InserterUtils.loadDelimitedDataTruth(insert, dir + "u" + fold + "-MultiCoreBPRMF.out");

        insert = data.getInserter(softmarginrankingmf_rating, evidencePartition2)
            InserterUtils.loadDelimitedDataTruth(insert, dir + "u" + fold + "-SoftMarginRankingMF.out");

        insert = data.getInserter(puresvd_rating, evidencePartition2)
            InserterUtils.loadDelimitedDataTruth(insert, dir + "u" + fold + "-pureSVD.out");

        insert = data.getInserter(wrmf_rating, evidencePartition2)
            InserterUtils.loadDelimitedDataTruth(insert, dir + "u" + fold + "-WRMF.out");

        println "after all baselines"

        // to predict
        insert = data.getInserter(rating, targetPartition2)
            InserterUtils.loadDelimitedDataTruth(insert, dir + "u" + fold + ".topredict");

        println "after topredict"

        //target partition
        Database db2 = data.getDatabase(targetPartition2, [user, item, rated,
                bprmf_rating,
                bprslim_rating,
                cofirank_rating,
                itemknn_rating,
                leastsquareslim_rating,
                mostpopular_rating,
                multicorebprmf_rating,
                softmarginrankingmf_rating,
                puresvd_rating,
                wrmf_rating
        ] as Set, evidencePartition2);

        //perform MPEInference
        //create the target partition


        //run MPE inference with learned weights
        MPEInference inferenceApp = new MPEInference(m, db2, config);
        MemoryFullInferenceResult inf_result = inferenceApp.mpeInference();

        if(inf_result.getTotalWeightedIncompatibility()!=null)
            println "[DEBUG inference]: Incompatibility = " + inf_result.getTotalWeightedIncompatibility()
                if(inf_result.getInfeasibilityNorm()!=null)
                    println "[DEBUG inference]: Infeasibility = " + inf_result.getInfeasibilityNorm()

                        inferenceApp.close();


        //keep track of the time
        TimeNow = new Date();
        println "after the inference time is " + TimeNow


            //call the garbage collector - just in case!
            System.gc();

        //Compute the RMSE
        HashMap<String, HashMap<String, Double>> users_items_ratings_labels = new HashMap<String, HashMap<String, Double>>();
        def labels = new File(dir + "u" + fold + ".test")
            def words, user, item, rating_value
            labels.eachLine {
                line ->
                    words = line.split("\t")
                    user=words[0].toString();
                item=words[1].toString();
                rating_value=words[2].toDouble();
                //user already exists
                if(users_items_ratings_labels.containsKey(user)){
                    HashMap<String, Double> items_ratings = users_items_ratings_labels.get(user)
                        items_ratings.put(item, rating_value)
                }
                else{	//first time to create an entry for this user
                    HashMap<String, Double> items_ratings = new HashMap<String, Double>()
                        items_ratings.put(item, rating_value)
                        users_items_ratings_labels.put(user, items_ratings)
                }
            }

        println "Inference results with weights learned from perceptron algorithm:"
            Double RMSE = 0.0
            Double MAE = 0.0
            int n=0
            int number_of_higher_predictions = 0;
        for (GroundAtom atom : Queries.getAllAtoms(db2, rating)){
            user = atom.arguments[0].toString()
                item = atom.arguments[1].toString()
                rating_predicted = atom.getValue().toDouble()
                //search in the structure users_items_ratings_labels for the pair <user,item>
                //and if it does exist then compute the RMSE error
                if(users_items_ratings_labels.containsKey(user)){
                    HashMap<String, Double> items_ratings = users_items_ratings_labels.get(user);
                    if(items_ratings.containsKey(item)){
                        rating_labeled = items_ratings.get(item);
                        println "( " + user + "," + item + " ) = " + rating_predicted + "\t" + rating_labeled;
                        RMSE += (rating_labeled-rating_predicted)*(rating_labeled-rating_predicted);
                        MAE += Math.abs(rating_labeled-rating_predicted);
                        if(rating_labeled < rating_predicted)
                            number_of_higher_predictions++;
                        //also count the number of ratings that the prediction is higher than the actual rating

                        n++;
                    }
                }
        }

        RMSE = Math.sqrt(RMSE/(1.0*n))
            totalRMSE += RMSE
            MAE = MAE/(1.0*n)
            totalMAE += MAE;
        per_higher_pred = (number_of_higher_predictions*1.0)/(1.0*n)
            println "number of higher predictions than the labeled " + number_of_higher_predictions
            println "percentage of higher predictions than the labeled " + per_higher_pred

            println "RMSE " + RMSE*5
            println "MAE " + MAE*5
            db2.close();

}//end folds

//compute the total RMSE and MAE

Double avgRMSE = totalRMSE/num_folds;
println "Avg RMSE = " + avgRMSE*5

Double avgMAE = totalMAE/num_folds;
println "Avg MAE = " + avgMAE*5

//keep track of the time
TimeNow = new Date();
println "end time is " + TimeNow


