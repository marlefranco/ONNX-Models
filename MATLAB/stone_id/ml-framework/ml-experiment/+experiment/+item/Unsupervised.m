classdef Unsupervised < experiment.item.Base
    %UNSUPERVISED Summary of this class goes here

    % Copyright 2021 The MathWorks Inc.
    
    properties 
        Label experiment.Label
    end
    
    methods
        function obj = Unsupervised(varargin)
            %UNSUPERVISED Construct an instance of this class
            
           obj = obj@experiment.item.Base( varargin{:} );
            
        end %function
        
        
        function values = initializeModelParams( obj )
                        
            %Initialize model settings
            definedInDataPipeline = obj.initialize();
            values = [definedInDataPipeline obj.modelsettings];

        end %function

        function value = items(obj)
            
            value = [obj.Label];
            
        end %function
        
 
         function [result, info] = select( obj, options )
            %SELECT TODO DOC
            
            arguments
                obj 
                options.Id (1,1) double {mustBeInteger}
                options.Model (1,1) double {mustBeInteger}
                options.Metadata (1,1) logical = false
            end
            
            if obj( options.Id ).Prepared == true && ~isempty(obj( options.Id ).Label)
            
                if options.Metadata == true

                    result = struct();
                    result.name = obj( options.Id ).describe.modelType( options.Model );
                    result.label = obj( options.Id ).Label( options.Model ).label;
                    result.trialId = options.Id;
                    result.labelId = options.Model;
                    
                    %Return Custom Properties in Data Preparation Pipeline
                    pipesettings = obj( options.Id ).Data.Properties.CustomProperties;
                    fields = string( fieldnames(pipesettings) );
                    
                    result.pipesettings = struct();
                    if ~isempty( fields )
                        for iField = fields(:)'
                            result.pipesettings.( iField ) = pipesettings.(iField);
                        end %for iField
                    end %if ~isempty( fieldnames )
                    
                    %Return property sweep 
                    result.pipelinesettings = obj( options.Id ).pipelinesettings;
                    result.modelsettings = obj( options.Id ).modelsettings;
                    
 %                   result.data = obj( options.Id ).result.Data;
                    info = obj( options.Id ).Label( options.Model ).describe();
                    
                else
                    
                    result  = obj( options.Id ).Label( options.Model ).label;
                    info    = obj( options.Id ).Label( options.Model ).describe();
                    
                end %options.Metadata
            
            else
                
                result = [];
                info = [];
            end
            
        end %function
          
    end %methods
    
    methods
        
        function reset(obj)

            obj.Label = experiment.Label.empty;
            
        end %reset 
        

        function value = initialize( obj )
            
            value = {...
                "FeatureNames", obj.features};
        end
        
         function result = describe( obj )
            %DESCRIBE
            
            result = [];
            for iObj = 1: numel(obj)
                
                thisReport = obj(iObj).Label.describe;
                result = [ result; thisReport  ]; %#ok<AGROW>
                
            end %iObj
            
        end %describe
        
        
        function result = lastfit( obj )
            %LASTMODEL 
            
            if ~isempty( obj.Label )
               result = obj.Label( end );
            else 
               result = [];  
            end
            
        end %lastfit
        
        function value = mdl( obj, index )     
            %MDL
            
            if isvalid( obj.Label(index) ) && ~isempty( obj.Label(index) )
                value =  obj.Label( index ).mdl;
            else
                value = [];
            end
        end %mdl
    
    end %from LabelPipeline
end %classdef 

