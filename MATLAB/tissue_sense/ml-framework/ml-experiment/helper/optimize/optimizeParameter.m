classdef (Abstract) optimizeParameter < matlab.mixin.SetGet & ...
        matlab.mixin.Heterogeneous
    %optimizeParameter Define parameters for experimental
    % design/configuaration
    %
    % Name/Value
    %   Name:   name of parameter in pipeline
    %
    %   Type: [ "Continuous" | "Integer" | "Discrete" | "Set" ]
    %
    %   Range:
    %       If Continuous or Integer: [LowerBound, UpperBound]
    %       If Discrete: [EntryA, EntryB, ...]
    %       If Set: { EntryA, EntryB }
    %
    %
    % Example: Continuous
    %
    %   Parameter = optimizeParameter.new(...
    %      "Name", "Example", ...
    %      "Range", [1 10],...
    %      "Type", "Continuous");
    %
    %
    % Example: Integer
    %
    %   Parameter = optimizeParameter.new(...
    %      "Name", "Example", ...
    %      "Range", [1 10],...
    %      "Type", "Integer");
    %
    %
    % Example: Discrete
    %
    %   Parameter = optimizeParameter.new(...
    %      "Name", "Example", ...
    %      "Range", [1 10],...
    %      "Type", "Discrete");
    %
    %
    % Example: Set
    %
    %   Parameter = optimizeParameter.new(...
    %      "Name", "Example", ...
    %      "Range", {[1 10],["A" "B"], struct()}...
    %      "Type", "Set");
    %
    %   N.Howes
    %   MathWorks Consulting 2020
    %
    
    properties
        Name (1,1) string
        Range (1,:)
        Optimize (1,1) logical = true
    end
    
    properties ( Dependent )
        Type
    end
    
    properties ( Hidden )
        Samples (1,1) double = 1
    end
    
    properties ( Dependent, Hidden )
        Method
    end
    
    properties ( Abstract, Access = protected )
        Method_
    end
    
    methods
        function obj = optimizeParameter( varargin )
            %optimizeBase Construct an instance of this class
            
            if nargin > 0
                
                if ~isempty( varargin )
                    set(obj, varargin{:})
                end
                
            end
        end
    end %methods
    
    methods
        function set.Method( obj, value )
            obj.Method_ = value;
        end
        
        function value = get.Method( obj )
            value = obj.Method_;
        end
        
        function value = get.Type(obj)
            value = string( class(obj) );
        end
    end %methods
    
    methods (Static)
        function value = new( options )
            %new TODO
            %
            
            arguments
                options.Name (1,1) string
                options.Range (1,:)
                options.Type (1,1) string ...
                    {mustBeMember(options.Type,["Continuous" "Integer" "Discrete" "Set"])}
            end
            
            switch options.Type
                case "Continuous"
                    value = optimizeContinuous( "Name", options.Name, ...
                        "Range", options.Range);
                case "Integer"
                    value = optimizeInteger( "Name", options.Name, ...
                        "Range", options.Range);
                case "Discrete"
                    value = optimizeDiscrete( "Name", options.Name, ...
                        "Range", options.Range);
                case "Set"
                    value = optimizeSet( "Name", options.Name, ...
                        "Range", options.Range);
            end %options Type
            
        end %function
        
        function values = listPipelineOptions( functionname )
            %listPipelineParameters Return a list of valid pipeline
            %configuation parameters for experimentation with the
            %framework.
            %
            % Syntax:
            %   values = listPipelineOptions( function ) where
            %   function is a function name or path to function.
            %
            
            arguments
                functionname (1,1) string
            end %arguments
            
            values = validatePipelineParameters( functionname );
            
        end %function
        
        function  values = listModelParameters( type )
            %listModelParameters Return a list of valid model
            %configuation parameters for experimentation with the
            %framework using autoML.
            %
            % Syntax:
            %   values = listModelParameters( type ) where type is
            %   experiment type: "Classification","Regression" ...
            %       "RUL", "SemiSupervised", or "Unsupervised"
            %
            
            arguments
                type (1,1) string ...
                    {mustBeMember(type, ["Classification","Regression" ...
                    "RUL", "SemiSupervised", "Unsupervised"])}
            end %arguments
            
            switch type
                case "Classification"
                    this = experiment.Classification();
                case "Regression"
                    this = experiment.Regression();
                case "RUL"
                    this = experiment.RUL();
                case "SemiSupervised"
                    this = experiment.SemiSupervised();
                case "Unsupervised"
                    this = experiment.Cluster();
            end %switch
            
            values = this.ValidModelParameters();
            
        end %function
        
        function values = defaultAutoMLOptimization( type, options )
            % TODO 
            
            arguments
                type (1,1) string ...
                    {mustBeMember(type, ["Classification","Regression" ...
                    "RUL", "SemiSupervised", "Unsupervised"])}
                options.Prototype (1,1) logical = false;
            end %arguments
            
            values = optimizeParameter.empty(0,1);
            
            if options.Prototype
                iterations = 3;
            else
                iterations = 30;
            end %if options.Prototype
            
            switch type
                case "Classification"
                    
                    values(1) = optimizeParameter.new(...
                        "Name", "Learners", ...
                        "Range", { ["tree", "discr", "nb", "knn",  "svm", ...
                        "ensemble", "ecoc", "linear", "kernel", "nnet"] },...
                        "Type", "Set");
                    
                    values(2) = optimizeParameter.new(...
                        "Name", "OptimizeHyperparameters", ...
                        "Range", "all",...
                        "Type", "Discrete");
                    
                    values(3) = optimizeParameter.new(...
                        "Name", "HyperparameterOptimizationOptions",...
                        "Range", {struct('MaxObjectiveEvaluations', iterations, ...
                        'UseParallel', true, 'ShowPlots', true)},...
                        "Type", "Set");
                    
                case "Regression"
                    
                    values(1) = optimizeParameter.new(...
                        "Name", "Learners", ...
                        "Range", { ["tree",  "svm",  "ensemble", ...
                        "gp", "linear", "kernel",  "nnet" "treebagger"] },...
                        "Type", "Set");
                    
                    values(2) = optimizeParameter.new(...
                        "Name", "OptimizeHyperparameters", ...
                        "Range", "all",...
                        "Type", "Discrete");
                    
                    values(3) = optimizeParameter.new(...
                        "Name", "HyperparameterOptimizationOptions",...
                        "Range", {struct('MaxObjectiveEvaluations', iterations, ...
                        'UseParallel', true, 'ShowPlots', true, 'MaxTime', 300)},...
                        "Type", "Set");
                    
                case "RUL"
                    
                    %TODO Confirm w/ Sudheer
                    values(1) = optimizeParameter.new(...
                        "Name", "Learners", ...
                        "Range", { ["linDegradation", "expDegradation"] },...
                        "Type", "Set");
                    
                case "SemiSupervised"
                    
                    values(1) = optimizeParameter.new(...
                        "Name", "Learners", ...
                        "Range", { ["graph", "self"] },...
                        "Type", "Set");
                    
                case "Unsupervised"
                    
                    values(1) = optimizeParameter.new(...
                        "Name", "Learners", ...
                        "Range", { ["hierarchical", "som", "kmeans", ...
                        "kmedoids", "gmm", "spectral" ] },...
                        "Type", "Set");
                    
            end %switch
            
        end %function
        
    end %methods
    
end %classdef



