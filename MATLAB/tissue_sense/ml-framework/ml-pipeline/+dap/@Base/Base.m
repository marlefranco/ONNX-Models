classdef ( Abstract )Base < matlab.mixin.SetGet
    %BASE (Abstract) BasePipeline
    
    %   Copyright 2019 The MathWorks, Inc.
    %
    % Auth/Revision:
    %   MathWorks Consulting
    %   $Author: nhowes $
    %   $Revision: 324 $  $Date: 2019-10-29 13:16:05 -0400 (Tue, 29 Oct 2019) $
    % ---------------------------------------------------------------------
    
    properties %(SetAccess = protected, GetAccess = public)
        Data table = table
    end
    
    properties ( Dependent )
        VariableNames
    end
    
    properties (Dependent)
        features
    end
    
    properties
        Name (1,1) string
    end
    
    properties ( Access = protected )
        Step (1,:) dap.stepInPipeline.Record
        Source table = table;
    end
    
    properties ( Access = protected )
        Prior table = table;
    end
    
    methods
        function varargout = apply( obj, functions, varargin )
            varargout       = cell(1, nargout);
            
            if nargin( functions ) == 1
                [varargout{:}]  = functions( obj.Data );
            else
                [varargout{:}]  = functions( obj.Data, varargin{:});
            end
        end %apply
        
        
        function value = table( obj )
            value = obj.Data;
        end %table
        
        
        function summary ( obj )
            summary( obj.Data )
        end %summary
        
        
        function head( obj, varargin )
            head(obj.Data, varargin{:} )
        end %head
        
        
        function tail( obj, varargin )
            tail(obj.Data, varargin{:} )
        end %tail
        
        
        function value = history( obj )
            value = {obj.Step.Operation}';
        end %history
        
        
        function reset( obj )
            obj.Data = obj.Source;
            obj.Step = dap.stepInPipeline.Record.empty(1,0);
        end %reset
        
        function resetmodel( obj )
            types = [obj.Step.Type];
            tF = ~ismember(types, "Data");
            obj.Step(tF) =[];
        end
        
        
        function undo( obj )
            
            if ~isempty( obj.Prior )
                obj.Data        = obj.Prior;
                obj.Step( end ) = [];
                obj.Prior       = [];
            else
                warning( "Nothing to undo" )
            end
            
        end %undo
        
        
        function delete( obj, stepIndex )
            
            obj.Step( stepIndex ) = [];
            steps = obj.Step;
            obj.reset()
            
            nSteps = numel( steps );
            for iStep = 1 : nSteps
                steps( iStep ).restore( obj )
            end
            
        end %delete
        
        
        function insert( obj, stepIndex, functions, typeOfFunction )
            
            if nargin < 4
                typeOfFunction = "custom" ;
            end
            
            steps = obj.Step;
            
            obj.reset()
            
            stepsPrior = 1 : stepIndex-1;
            for iStep = stepsPrior
                steps( iStep ).restore( obj )
            end
            
            switch lower( typeOfFunction )
                case "custom"
                    obj.update( func )
                case "framework"
                    functions( obj )
            end
            
            stepsPost = stepIndex : numel(steps);
            for iStep = stepsPost
                steps( iStep ).restore( obj )
            end
            
        end %insert
        
        
        function value = export( obj )
            value = obj.Step;
        end %export
        
        
        function save( obj, location )
            
            if nargin < 2
                error( "Please provide a filename to save file." )
            end
            
            name = obj.Name;
            pipelineSteps = obj.export();
            save( location, 'pipelineSteps', 'name' )
            
        end %save
        
    end %public
    
    
    methods
        function value = get.VariableNames( obj )
            value = string( obj.Data.Properties.VariableNames );
        end %function
 
        function value = get.features( obj )
            
            if ~isempty( obj.Data )
                
                if contains( "Features", properties(obj.Data.Properties.CustomProperties) )
                    value = obj.Data.Properties.CustomProperties.Features;
                else
                    warning('Custom properties not defined in data pipeline. \n%s', ...
                        'Setting default values for features and response');
                    if contains( "Partition", obj.Data.Properties.VariableNames )
                        value = string( obj.Data.Properties.VariableNames( 1:end-2 ) );              
                        obj.Data = addprop(obj.Data, "Features", "table" );
                        
                        obj.Data.Properties.CustomProperties.Features = value;
                    else
                        value = string( obj.Data.Properties.VariableNames( 1:end-1 ) );
                        obj.Data = addprop(obj.Data, "Features", "table" );
                        
                        obj.Data.Properties.CustomProperties.Features = value;
                    end
                end
                
            else
                value = "";
            end
            
        end %function
        
    end %get
end %classdef

