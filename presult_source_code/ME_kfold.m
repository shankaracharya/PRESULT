function ME_kfold(varargin)
fprintf('\n Example command line: "./presult kfold ME --input_data <input_data.txt> --randm <yes/no> --kfold <int>" \n')
p = inputParser;
addParameter(p,'input_data','pid_raw_data.txt',@ischar);
addParameter(p,'csv','no',@ischar);
addParameter(p,'missing_value','?');
addParameter(p,'kfold','10');
addParameter(p,'emiters','7');
addParameter(p,'var','variables.txt',@ischar);
addParameter(p,'actfunc','logistic',@ischar);
addParameter(p,'out','ME',@ischar);
addParameter(p,'hexp','8');
addParameter(p,'hgate','20');
addParameter(p,'nexp','2');
addParameter(p,'mstep_iters','7');
addParameter(p,'randm','shuffle',@ischar);
addParameter(p,'title','Mixture of Experts',@ischar);

new_array = varargin(:);
for i =1:size(new_array,1)
     for j = 1:size(p.Parameters,2)
         param=p.Parameters{j};
         array_entry=new_array{i};
         first_check=strcat('-',param);
         second_check=strcat('--',param);
         if strcmp(array_entry,first_check)
            new_array{i}=param;
         end
         if strcmp(array_entry,second_check)
            new_array{i}=param;
         end
     end
end         

new_array(strcmp('yes',new_array))={'shuffle'};
new_array(strcmp('no',new_array))={'default'};

parse(p,new_array{:});


data_file = (p.Results.input_data);
csv = (p.Results.csv);
miss = (p.Results.missing_value);
var = (p.Results.var);
kvalue = (p.Results.kfold);
n = (p.Results.emiters);
actfunc = (p.Results.actfunc);
hexp = (p.Results.hexp);
hgate = (p.Results.hgate);
nexp = (p.Results.nexp);
out = (p.Results.out);
ran = (p.Results.randm);
mstep_iters = (p.Results.mstep_iters);
title_me = (p.Results.title);
k = str2double(kvalue);
n = str2double(n);
hexp = str2double(hexp);
hgate = str2double(hgate);
nexp = str2double(nexp);
mstep_iters = str2double(mstep_iters);

if strcmp(csv,'no') && strcmp(miss,'?') 
   perl('newest_normalization.pl', '--input', data_file, '--output', 'normalized_data_kfold_ME.txt','--var',var);
else
   perl('newest_normalization.pl', '--input', data_file, '--output', 'normalized_data_kfold_ME.txt', '--csv','--var',var);
end


data_file1 = load('normalized_data_kfold_ME.txt');

rownumbers = size(data_file1,1);
%%%%%%%%%%%%%%%%%%%%
%DataSorting start
%%%%%%%%%%%%%%%%%%%%
rng(ran);
ran = rand(rownumbers,1);
data_file2 = [ran data_file1];
data_file3 = sortrows(data_file2,1);
data_file = data_file3(:,2:end);
%%%%%%%%%%%%%%%%%%%%%
%Data Sorting end
%%%%%%%%%%%%%%%%%%%%%%

chunk = floor(rownumbers/k);

hold all;

grid on;

ah1 = gca;

    for f = 1:k
        fprintf('\n ############ %d-fold cross validation ############### \n',f);
        test = data_file((f-1)*chunk+1:(f)*chunk,:);
        train = [data_file(1:(f-1)*chunk,:); data_file(f*chunk+1:end, :)];
        save ([out,'_train_data_kfold_',int2str(f),'.txt'],'train','-ascii');
        save ([out,'_test_data_kfold_',int2str(f),'.txt'],'test','-ascii');
        q = load([out,'_train_data_kfold_',int2str(f),'.txt']);
        ninput = size(q,2);
        x = q(:,1:ninput-1);
        t = q(:,ninput);
        net =  mixmlp(ninput-1,hexp,1,hgate,nexp,actfunc,'standard');
        options  = zeros(1,18);
        options(1)  =  -1;  % no display of the error value within the M-step 
        options(14) =  mstep_iters;  % the number of iterations in the M-step   
        [net,~] = mixmlpem( net, x, t, options, n,'scg');
        save ([out,'_trained_net_kfold_',int2str(f)],'net');
        fprintf(['\n Trained network for ', int2str(f), '-kfold ', 'ME Model is saved as: ']);
        fprintf([out,'_trained_net_kfold_',int2str(f),'.mat'],'\n');
        fprintf('\n');
        test = load([out,'_test_data_kfold_',int2str(f),'.txt']);
        ninput = size(test,2);
        xtest = test(:,1:ninput-1);
        ttest = test(:,ninput);
        y_test = mixmlpfwd(net,xtest);
        [C]=confmat(y_test,ttest);
        fprintf('\n Confusion Matrix of the Test Model: \r\n');
        disp(C);
        C1 = C(1,1);
        C2 = C(1,2);
        C3 = C(2,1);
        C4 = C(2,2);
        Correct_prediction = C1+C4;
        Wrong_prediction = C2+C3;
        Sensitivity = C1/(C1+C3);
        Specificity = C4/(C4+C2);
        Accuracy = (Correct_prediction/(Correct_prediction + Wrong_prediction))*100;
        fprintf(['\n Train data for ', int2str(f), '-kfold ', 'is saved as: ']);
        fprintf([out,'_train_data_kfold_',int2str(f),'.txt'],'\n');
        fprintf('\n');
        fprintf(['\n Test data for ', int2str(f), '-kfold ', 'is saved as: ']);
        fprintf([out,'_test_data_kfold_',int2str(f),'.txt'],'\n');
        fprintf('\n');
        fprintf('\n Sensitivity of the Test Model = %f\r', Sensitivity);
        fprintf('\n Specificity of the Test Model = %f\r', Specificity);
        fprintf('\n Accuracy of the Test Model = %f\r', Accuracy);
%         [x_ME_train,y_ME_train,~,AUC_ME_train] = perfcurve(t,y_train',size(t,2));
%         [x_ME_test,y_ME_test,~,AUC_ME_test] = perfcurve(ttest,y_test',size(t,2));
        [faRate, hitRate, AUC_ME_test] = rocCurve(y_test, ttest);
%         [faRate, hitRate, AUC_ME_train] = rocCurve(y_train, t);
%         fprintf('\n AUC of the Train Model = %f\r\n', AUC_ME_train);
        fprintf('\n AUC of the Test Model = %f\r\n', AUC_ME_test);
        legendinfo{f} = ['AUC = ' num2str(AUC_ME_test)];
        AUC_COL_test(f,:) = AUC_ME_test;
%         fprintf('\n ############ End of %d-fold cross validation ############### \n',f);
%         AUC_COL_train(f,:) = AUC_ME_train;
        ACCU(f,:) = Accuracy;
        SENS(f,:) = Sensitivity;
        SPEC(f,:) = Specificity;
        kfold(f,:) = f;
        
    end
    hold off;
    Avg_AUC_test = mean(AUC_COL_test);
    legend(ah1,legendinfo,'Location','SouthEast','FontSize',14);
    xlabel_kf = xlabel(ah1,'1-Specificity');
    set(xlabel_kf,'FontSize',16);
    ylabel_kf = ylabel(ah1, 'Sensitivity');
    set(ylabel_kf,'FontSize',16);
    set(gcf,'color','w');
    set(gca,'FontSize',16);
    box on;
    dim =[0.63 0.4 0.08 0.2];
    str = ['Avg AUC = ',num2str(Avg_AUC_test)];
    annotation('textbox',dim,'String',str,'FitBoxToText','on','FontSize',14,'Color','k');
    title_kf = title(sprintf(title_me));
    set(title_kf,'FontSize',16);
    Average_sesitivity = mean(SENS);
    Average_specificity = mean(SPEC);
    Average_accuracy = mean(ACCU);
    fprintf('\n ############ Summary for %d-fold cross validation ############### \n',k);
    summary = [kfold SENS SPEC ACCU AUC_COL_test];

    TT = array2table(summary,'VariableNames',{'kfold' 'Sensitivity' 'Specificity' 'Accuracy' 'AUC_test'});
    disp(TT);
    fprintf('\n Average Accuracy of the Test Model = %f\r\n', Average_accuracy);
    
%     Avg_AUC_train = mean(AUC_COL_train);
%     title_kf = title(ah1, sprintf('ME Avg Auc=%5.3f', Avg_AUC_test));
%     set(title_kf,'FontSize',16);
    fprintf('\n Average Sensitivity of the Test Model = %f\r\n', Average_sesitivity);
    fprintf('\n Average Specificity of the Test Model = %f\r\n', Average_specificity);
    fprintf('\n Average AUC of the Test Model = %f\r\n', Avg_AUC_test);
%     fprintf('\n Average AUC of the Train Model = %f\r\n', Avg_AUC_train);
    kfold_out = [out,'_kfold_ROC'];
    export_fig (kfold_out, '-pdf', '-eps');
    
end