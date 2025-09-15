classdef util
    % OPS.UTIL Utilities for the Olympus project
    %
    % ops.util methods:
    %   import - Import data
    %   getBasicStats -
    %
    
    methods (Static)
        function value = import( options )
            
            arguments
                options.Type (1,1) string {mustBeMember(options.Type, ...
                    ["normal", "classD", "challenging", "DiffSource", "2-class"])} = "normal" ;
            end
            
            %IMPORT Import data
            switch options.Type
                case "normal"
                    T = readtable( "OSTA Data for Mathworks.xlsx", ...
                        'VariableNamingRule', 'modify' );
                    T(:,1) = [];
                    T = rows2vars(T);
                    
                case "classD"
                    T = readtable( "OSTA Data for Mathworks_Class D data.xlsx", ...
                        'VariableNamingRule', 'modify');
                    T(:,1) = [];
                    T = rows2vars(T);
                    
                case "challenging"
                    T = readtable( "OSTA Data for Mathworks_classification problem.xlsx", ...
                        'VariableNamingRule', 'modify', ...
                        'Sheet' , "Spectra");
                    T(:,1) = [];
                    T = rows2vars(T);
                    
                case "DiffSource"
                    T = readtable( "OSTA_Data_for_Mathworks_different_source.xlsx", ...
                        'VariableNamingRule', 'modify');
                    T(:,1) = [];
                    T = rows2vars(T);
                    
                case "2-class"
                    class1 = readtable( "OSTA_Data_for_Mathworks_2classes.xlsx", ...
                        'VariableNamingRule', 'modify', ...
                        'Sheet', "Class1");
                    class2 = readtable( "OSTA_Data_for_Mathworks_2classes.xlsx", ...
                        'VariableNamingRule', 'modify', ...
                        'Sheet', "Class2");
                    
                    class1(:,1) = [];
                    class2(:,1) = [];
                    T = [rows2vars(class1); rows2vars(class2)];
                    T.OriginalVariableNames(1:width(class1)) = {'Class1'};
                    T.OriginalVariableNames(width(class1)+1:width(class1)+width(class2)) = {'Class2'};
            end
            
            T = movevars(T, 'OriginalVariableNames', 'After', width(T));
            
            T.OriginalVariableNames = string(T.OriginalVariableNames);
            T.OriginalVariableNames(contains(string(T.OriginalVariableNames), "ClassA")) = "ClassA";
            T.OriginalVariableNames(contains(string(T.OriginalVariableNames), "ClassB")) = "ClassB";
            T.OriginalVariableNames(contains(string(T.OriginalVariableNames), "ClassC")) = "ClassC";
            T.OriginalVariableNames(contains(string(T.OriginalVariableNames), "ClassD")) = "ClassD";
            T.OriginalVariableNames = categorical(T.OriginalVariableNames);
            
            T.Properties.VariableNames(end) = "response";
            
            value = T;
            
        end %function
        
        function [T, featurenames] = getBasicStats(X, options)
            
            arguments
                X table
                options.ResponseName (1,1) string = X.Properties.VariableNames(end)
                options.FeatureNames (1,:) string = X.Properties.VariableNames(1:end-1);
            end
            
            value = X(:, options.FeatureNames);
            
            fcnArray{1} = @(x)mean(x, 2 );
            fcnArray{2} = @(x)range(x, 2);
            fcnArray{3} = @(x)iqr(x, 2);
            fcnArray{4} = @(x)std(x, [], 2);
            fcnArray{5} = @(x)min(x, [], 2);
            fcnArray{6} = @(x)max(x, [], 2);
            fcnArray{7} = @(x)mad(x, 0, 2);
            fcnArray{8} = @(x)skewness(x, 1, 2);
            fcnArray{9} = @(x)kurtosis(x, 1, 2);
            fcnArray{10} = @(x)prctile(x, 10, 2);
            fcnArray{11} = @(x)median(x, 2);
            fcnArray{12} = @(x)prctile(x, 90, 2);
            
            T = table('Size', [size(value,1), 12], ...
                'VariableTypes', repmat("double", 1, 12), ...
                'VariableNames', ["Mean", "Range", "IQR", "Std", "Min", "Max", "Mad", "Skewness", "Kurtosis", "Prctile10", "Median", "Prctile90"]);
            
            for i = 1:length(fcnArray)
                T{:,i} = fcnArray{i}(value.Variables);
            end
            
            featurenames = T.Properties.VariableNames;
            T.(options.ResponseName) = X.(options.ResponseName);
            
        end
        
        function plotSummarySignals(data, responseVar, options)

            arguments
                data table
                responseVar (1,1) string = "Response"
                options.predictorNames (1,:) string = "";
            end

            response = categorical(data.(responseVar));

            if strcmp(options.predictorNames, "")
                predictorNames = string(data.Properties.VariableNames(...
                    contains(data.Properties.VariableNames, "Feature")))';
            else
                predictorNames = options.predictorNames;
            end
            
            figure;
            p = plot(data{:,predictorNames}');

            uniRes = unique(response);
            catIdx = zeros(length(uniRes),1);
            c = lines(length(uniRes));
            for k=1:length(uniRes)
                thisResIdx = response==uniRes(k);
                arrayfun(@(x) set(x, 'Color', c(k,:)), p(thisResIdx));
                catIdx(k) = find(thisResIdx,1);
            end
            legend(p(catIdx), uniRes);
                
        end % plotSummarySignalsByFile

    end %methods
    
    
end %classdef

