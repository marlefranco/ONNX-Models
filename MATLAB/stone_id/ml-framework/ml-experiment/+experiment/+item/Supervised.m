classdef Supervised < experiment.item.Base
    %SUPERVISED Summary of this class goes here

    % Copyright 2021 The MathWorks Inc.
            
    properties (Dependent)
        categoricalfeatures
        response
        trainingobs
        testobs
    end
    
    methods
        function obj = Supervised(varargin)
            %SUPERVISED Construct an instance of this class

           obj = obj@experiment.item.Base( varargin{:} );
            
        end %function
 
        function values = initializeModelParams( obj, type )
            
            arguments
               obj
               type (1,1) string ...
                   {mustBeMember(type, ["standard" "pdm"])} = "standard"
            end %arguments 
            
            %Initialize model settings
            switch type
                case "standard"
                    definedInDataPipeline = obj.initialize();
                    values = [definedInDataPipeline obj.modelsettings];
                case "pdm"
                    definedInDataPipeline = obj.initialize( "pdm" );
                    values = [definedInDataPipeline obj.modelsettings];
                otherwise
                    error("Unhandled type initialize option.")               
            end %switch type
            
        end %function
        
        
        function value = items(obj)

            value = [obj.model];
            
        end %function

    end %methods
    
    methods

        function value = initialize( obj, type )
            
            arguments
                obj
                type (1,1) string ...
                    {mustBeMember(type, ["standard" "pdm"])} = "standard"
            end %arguments
            
            switch type
                case "standard"
                    value = {...
                        "PredictorNames", obj.features, ...
                        "ResponseName", obj.response, ...
                        "Include", obj.trainingobs};
                case "pdm"
                    value = {...
                        "HealthIndicatorName", obj.Data.Properties.CustomProperties.HealthIndicatorName, ...
                        "DataVariable", obj.Data.Properties.CustomProperties.DataVariable, ...
                        "LifeTimeVariable", obj.Data.Properties.CustomProperties.LifeTimeVariable};
            end %switch
            
        end %function

        function result = describe( obj )
            %DESCRIBE
            
            result = [];
            for iObj = 1: numel(obj)
                
                thisReport = obj(iObj).model.describe;
                result = [ result; thisReport ]; %#ok<AGROW>
                
            end %iObj
            
        end %describe
        
        
        function result = lastmodel( obj )
            %LASTmodel
            
            if ~isempty( obj.model )
                result = obj.model( end );
            else
                result = [];
            end
            
        end %lastmodel
        
        function value = mdl( obj, index )
            %MDL
            
            if isvalid( obj.model(index) ) && ~isempty( obj.model(index) )
                value =  obj.model( index ).mdl;
            else
                value = [];
            end
        end %mdl
        
        function reset(obj)

            obj.model = [];
            
        end %reset 
        
    end %from pipelines
    
    methods
        
        function value = get.response( obj )
            
            if ~isempty( obj.Data )
                
                if contains( "Response", properties(obj.Data.Properties.CustomProperties) )
                    value = obj.Data.Properties.CustomProperties.Response;
                else
                    
                    if contains( "Partition", obj.Data.Properties.VariableNames )
                        value = string( obj.Data.Properties.VariableNames( end-1 ) );
                        %Store Response, Features, and Partition
                        obj.Data = addprop(obj.Data, "Response", "table" );
                        
                        obj.Data.Properties.CustomProperties.Response = value;
                        
                    else
                        value = string( obj.Data.Properties.VariableNames( end ) );
                        obj.Data = addprop(obj.Data, "Response", "table" );
                        
                        obj.Data.Properties.CustomProperties.Response = value;
                    end
                end
                
            else
                value = "";
            end
            
        end %response
        
        function value = get.trainingobs( obj )
            
            if ~isempty( obj.Data )
                
                if contains("Partition", obj.Data.Properties.VariableNames)
                    value = obj.Data.Partition == "Train";
                else
                    value = true(height(obj.Data), 1);
                end
                
            else
                value = [];
            end
            
        end %trainingobs
        
        function value = get.testobs( obj )
            
            if ~isempty( obj.Data )
                
                if contains("Partition", obj.Data.Properties.VariableNames)
                    value = obj.Data.Partition == "Test";
                else
                    value = false(height(obj.Data), 1);
                end
                
            else
                value = [];
            end
            
        end
        
        function value = get.categoricalfeatures( obj )
            if obj.Features ~=""
                tF = varfun(@iscategorical, obj.Data, ...
                    'InputVariables', obj.features, ...
                    'OutputF', 'uni');
                value = obj.Features( tF );
            else
                value = [];
            end
        end %categoricalfeatures      
        
    end %get/set
    
    
end %classdef 

