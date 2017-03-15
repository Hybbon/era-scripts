cat("-- reading arguments\n", sep = "");
cmd_args = commandArgs();
arq <- tail(cmd_args,n=1) #pega a ultima posicao de um vetor


#read the data matrix
dados <- read.csv(arq, header=T,sep=';');

#count the number of metrics to be evaluated
num_metrics = ncol(dados) - 1
sample_size = nrow(dados)/length(levels(dados$alg))
#for each metric in data
algorithms = levels(dados$alg)
num_algs = length(algorithms)
for (metric in c(1:num_metrics)){
    
    #convert the metric index to "v#" label
    metric_index = paste('v',toString(metric),sep='') #We need to specify that the separator is the empty string
    #take the algorithms names in the colunm alg
    
    
    #take the values of the current metric to perform friedman test
    dados_aux <- matrix(dados[,metric_index],c(sample_size,num_algs))

    if (num_algs == 2){
        end_vet = length(dados[,metric_index])
        #print(dados[(sample_size+1):end_vet,metric_index])
        res = wilcox.test(dados[1:sample_size,metric_index], dados[(sample_size+1):end_vet,metric_index], p.adj="holm", paired=F)
        print(res$p.value)
        
    }

    res_friedman = friedman.test(dados_aux)

    if (res_friedman['p.value'] <= 0.05){

        
        #attach(dados)

        print(c("Statistical diff for metric",metric_index))
        #run wilconxon test for the values of the current metric
        res_wilcoxon = pairwise.wilcox.test(dados[,metric_index], dados[,'alg'], p.adj="holm", paired=F)

        #print(res_wilcoxon)
        for (ro in rownames(res_wilcoxon$p.value)){

            for (co in colnames(res_wilcoxon$p.value)){

                if (!is.na(res_wilcoxon$p.value[ro,co])){
                    
                    if (res_wilcoxon$p.value[ro,co] <= 0.05)
                        print(c(co,ro,res_wilcoxon$p.value[ro,co]))
                }

            }
        }

        #cat ("Pareado")
        #pairwise.wilcox.test(values, alg, p.adj="bonferroni", exact=F, paired = T)

        #detach(dados)

    }
}
