classdef fitdegradation
    
    methods (Static)
        
        function [mdl, info] = auto(data, custom, options)
            
            arguments
                data table
                custom.Learners (1,:) string ...
                    {mustBeMember(custom.Learners,["all", "linDegradation", "expDegradation"])} = "all";
                custom.Metric (1,1) string ...
                    {mustBeMember(custom.Metric, ["r2OnTrain","rmseOnTrain"])} = "rmseOnTrain"

                options.HealthIndicatorName (1,1) string
                options.DataVariable (1,1) string
                options.LifeTimeVariable (1,1) string
                options.UseParallel = false
            end
                       
            %Optional Name/Value
            args = namedargs2cell( options );
            
            infos = [];
            if any( matches( custom.Learners, ["all", "linDegradation" ] ) )
                [mdlLIN, infoLIN] = fitdegradation.linDegradation( data, args{:} );
                infos = vertcat(infos, infoLIN);
            end
            
            if any( matches( custom.Learners, ["all", "expDegradation" ] ) )
                [mdlEXP, infoEXP] = fitdegradation.expDegradation( data, args{:} );
                infos = vertcat(infos, infoEXP);
            end
            
            value = sortrows( infos, custom.Metric );
            
            switch extractBetween( value.modelType(1),"(",")" )
                case "expDegMdl"
                    mdl = mdlEXP; info = infoEXP;
                case "linDegMdl"
                    mdl = mdlLIN; info = infoLIN;
            end %switch case
            
        end
        
        
        function [mdl, info] = expDegradation(data, custom, options)
            
            arguments
                data table
                custom.HealthIndicatorName (1,1) string
                custom.DataVariable (1,1) string
                custom.LifeTimeVariable (1,1) string
                
                options.Theta (1,1) double = 1
                options.ThetaVariance (1,1) double {mustBeNonnegative} = 1e6
                options.Beta (1,1) double = 1
                options.BetaVariance (1,1) {mustBeNonnegative} = 1e6
                options.Rho (1,1) double {mustBeInRange(options.Rho,[-1,1])} = 0
                options.Phi (1,1) double = -1
                options.NoiseVariance (1,1) double {mustBeNonnegative} = 1
                options.SlopeDetectionLevel (1,1) double {mustBeInRange(options.SlopeDetectionLevel,[0,1])} = 0.05
                options.UseParallel = false
                options.LifeTimeUnit (1,1) string = ""
            end
            
            warning('off','predmaint:analysis:warnCovEstEnsembleOnlyOneMember')
            warning('off','predmaint:analysis:warnThetaMightCrossZero')
            
            %Optional Name/Value
            args = namedargs2cell( options );
            
            data = data.( custom.HealthIndicatorName );
            
            if ~iscell(data)
                error("Result of training data pipeline must be a table of cell arrays")
            end         
            
            mdl = exponentialDegradationModel( args{:} );

            fit( mdl, data, custom.LifeTimeVariable, custom.DataVariable )
            
            modelType = "Exponential Degradation (expDegMdl)";
            
            r2array = zeros(length(data),1);
            rmsearray = zeros(length(data),1);
            
            for ii = 1:length(data)
                dataTemp = data{ii};
                
                if istimetable(dataTemp)
                    dataTemp = timetable2table(dataTemp);
                end       
                
                x = (1:height(dataTemp))';
                y = dataTemp{:,2};
                
                [~, gof] = fit(x,y,'poly2');
                r2array(ii) = gof.rsquare;
                rmsearray(ii) = gof.rmse;
            end
            
            r2OnTrain = mean(r2array);
            rmseOnTrain = mean(rmsearray);
            
            info = table( modelType, r2OnTrain, rmseOnTrain );
            
        end %exponentialDegradation
        
        function [mdl, info] = linDegradation(data, custom, options)
            
            arguments
                data table
                custom.HealthIndicatorName (1,1) string
                custom.DataVariable (1,1) string
                custom.LifeTimeVariable (1,1) string
                
                options.Theta (1,1) double = 1;
                options.ThetaVariance (1,1) double {mustBeNonnegative} = 1e6
                options.Phi (1,1) double = -1
                options.NoiseVariance (1,1) double {mustBeNonnegative} = 1
                options.SlopeDetectionLevel (1,1) double {mustBeInRange(options.SlopeDetectionLevel,[0,1])} = 0.05
                options.UseParallel = false
                options.LifeTimeUnit (1,1) string = ""
            end
            
            warning('off','predmaint:analysis:warnCovEstEnsembleOnlyOneMember')
            warning('off','predmaint:analysis:warnThetaMightCrossZero')
            
            %Optional Name/Value
            args = namedargs2cell( options );
            
            data = data.( custom.HealthIndicatorName );
            
            if ~iscell(data)
                error("Result of training data pipeline must be a table of cell arrays")
            end
            
            mdl = linearDegradationModel( args{:} );
                        
            fit( mdl, data, custom.LifeTimeVariable, custom.DataVariable )
            
            modelType = "Linear Degradation (linDegMdl)";
            
            r2array = zeros(length(data),1);
            rmsearray = zeros(length(data),1);
            
            for ii = 1:length(data)
                dataTemp = data{ii};
                
                if istimetable(dataTemp)
                    dataTemp = timetable2table(dataTemp);
                end       
                
                x = (1:height(dataTemp))';
                y = dataTemp{:,2};
                
                [~, gof] = fit(x,y,'poly1');
                r2array(ii) = gof.rsquare;
                rmsearray(ii) = gof.rmse;
            end
            
            r2OnTrain = mean(r2array);
            rmseOnTrain = mean(rmsearray);
            
            info = table( modelType, r2OnTrain, rmseOnTrain );
            
        end %linearDegradation
        
    end %public
end %classdef

% Custom validation function
function mustBeInRange(arg,b)
if any(arg(:) < b(1)) || any(arg(:) > b(2))
    error(['Value assigned to Data is not in range ',...
        num2str(b(1)),'...',num2str(b(2))])
end
end