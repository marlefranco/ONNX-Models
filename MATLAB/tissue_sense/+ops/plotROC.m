function plotROC(AUC)

arguments
    AUC table
end

rocInfo = AUC.ROCcurve;
classnames = string(fieldnames(rocInfo));

lgdNames = repmat("", length(classnames), 1);
newVarNames = matlab.lang.makeValidName(AUC.Properties.VariableNames);

figure;
hold on;
for i = 1:length(classnames)
    plot(rocInfo.(classnames(i)).X, rocInfo.(classnames(i)).Y);
    lgdNames(i) = classnames(i) + " (AUC: " + AUC{1,newVarNames==classnames(i)} + ...
        ")";
end
hold off;
legend(lgdNames);

ylabel("True Positive Rate")
xlabel("False Positive Rate")

title('Receiver operating characteristic (ROC) curve');
end