classdef util < matlab.unittest.TestCase
    %UTIL Summary of this class goes here
    
    properties
        cardata
    end
    
    
    methods (TestClassSetup)
        function initialize( testCase )
            load carbig %#ok<LOAD>
            testCase.cardata = table(Acceleration, Cylinders, Displacement,...
                Horsepower, Model_Year, Weight, Origin, MPG);
        end %function
    end %methods
    
    methods (Test)
        
        function name( testCase )
            
            %Default
            value = util.name( testCase.cardata );
            testCase.verifyNumElements( value, 8 )
            testCase.verifyClass( value, "string" )
            
            %Valid condition
            tF = false(8,1); tF(2) = true;
            value = util.name( testCase.cardata, "tF", tF );
            testCase.verifyEqual( value, "Cylinders" )
            
            %Corner case
            value = util.name( testCase.cardata, "tF", false(8,1) );
            testCase.verifyEmpty( value )
            
            %Error condition: bad required
            testCase.verifyError( ...
                @()util.name( "A" ), ...
                'util:mustBeClass')
            
            %Error condition: bad NV (type)
            testCase.verifyError( ...
                @()util.name(testCase.cardata, "tF", "A"), ...
                'MATLAB:validation:UnableToConvert')
            
            %Error condition: bad NV (size) 
            testCase.verifyError( ...
                @()util.name(testCase.cardata, "tF", [true false]), ...
                'UTIL:NAME:wrongsize' )
            
        end %function
        
        function custom( testCase )
            
            %Default
            value = util.custom( testCase.cardata );
            testCase.verifyEmpty( properties(value) )
   
            %Error condition: bad required 
            testCase.verifyError( ...
                @()util.custom( "A" ), ...
                'util:mustBeClass')
            
            %Error condition: bad NV (type)
            testCase.verifyError( ...
                @()util.custom(testCase.cardata, 10), ...
                'util:mustBeClass')
            
            %TODO Add conditions 
            
        end %function
        
        function normalize( testCase )
            
           %Default
           value = util.normalize( testCase.cardata );
           testCase.verifyEqual( width(value), width(testCase.cardata) )
           
           %Valid condition 
           [value, info] = util.normalize( testCase.cardata, "zscore" );
           
           testCase.verifyTrue(...
               info{"Horsepower","Value"}(1) - nanmean(testCase.cardata.Horsepower) < 1e-6 ...
               )

           testCase.verifyTrue(...
               info{"Horsepower","Value"}(2) - nanstd(testCase.cardata.Horsepower) < 1e-6 ...
                )
           
            chkvalue = (testCase.cardata.Horsepower - nanmean(testCase.cardata.Horsepower)) ./ ...
                nanstd(testCase.cardata.Horsepower);
            
            testCase.verifyTrue(...
                all(value.Horsepower-chkvalue < 1e-6 | isnan( value.Horsepower-chkvalue) ) ...
                )
            
           %Valid condition 
           [value, info] = util.normalize( testCase.cardata, "range" );
            
           testCase.verifyEqual(...
               info{"Horsepower","Value"}(1), ...
               min(testCase.cardata.Horsepower) )
           
           testCase.verifyEqual(...
               info{"Horsepower","Value"}(2), ...
               max(testCase.cardata.Horsepower) )
           
           chkvalue = (testCase.cardata.Horsepower -  min(testCase.cardata.Horsepower) ) ./ ...
               (max(testCase.cardata.Horsepower) - min(testCase.cardata.Horsepower));
           
           testCase.verifyTrue(...
               all(value.Horsepower-chkvalue < 1e-6 | isnan( value.Horsepower-chkvalue) ) ...
               )
            
            %Valid condition 
           [~, info] = util.normalize( testCase.cardata, "auto" );
           
           tF = util.isnormal( testCase.cardata );
           testCase.verifyTrue( all(info.Mthd(tF) == "range") )
           
           %TODO Add conditions 
           %Synthetic normal 
           
           
        end %function
        
        function scaler( testCase )
            
            
            %Valid condition: zscore 
            [gtruth, info] = util.normalize( testCase.cardata, "zscore" );
            
            value = util.scaler( testCase.cardata, info );
            
            tF = util.isnumeric( gtruth );
            
            testCase.verifyTrue(...
                all( ...
                    all( gtruth(:,tF).Variables - value(:,tF).Variables < 1e-6 | ...
                    isnan( gtruth(:,tF).Variables ) ) ...
                    )...
                )

            %Valid condition: range
            [gtruth, info] = util.normalize( testCase.cardata, "range" );
            
            value = util.scaler( testCase.cardata, info );
            
            tF = util.isnumeric( gtruth );
            
            testCase.verifyTrue(...
                all( ...
                all( gtruth(:,tF).Variables - value(:,tF).Variables < 1e-6 | ...
                isnan( gtruth(:,tF).Variables ) ) ...
                )...
                )
            
            
            %Valid condition: mixed range and zscore 
            [gtruth1, info1] = util.normalize( testCase.cardata, "zscore", ...
                "DataVariables", ["Acceleration" "Cylinders" "Displacement" "Horsepower" "Model_Year"]);
            
            [gtruth2, info2] = util.normalize( testCase.cardata, "range", ...
                "DataVariables", ["Weight", "MPG"]);
            

            info = vertcat(info1, info2);
            gtruth = horzcat( gtruth1(:,info1.Properties.RowNames), ...
                gtruth2(:,info2.Properties.RowNames) );
            
            gtruth=addvars(gtruth,testCase.cardata.Origin, 'Before', "MPG");    
            
            
            value = util.scaler( testCase.cardata, info );
            
            tF = util.isnumeric( gtruth );
            
            testCase.verifyTrue(...
                all( ...
                    all( gtruth(:,tF).Variables - value(:,tF).Variables < 1e-6 | ...
                    isnan( gtruth(:,tF).Variables ) ) ...
                    )...
                )
            
            
        end %function
        
        function log10( testCase )
            
        end %function
        
        function pow10( testCase )
            
        end %function
        
        function skewness( testCase )
            
        end %function
        
        function descriptivestatistics( testCase )
        
        end %function
        
        function summarize( testCase )
            
        end %function
        
        function isvar( testCase )
            
            value = util.isvar( testCase.cardata, "Acceleration" );
            testCase.verifyTrue( value )
            
            value = util.isvar( testCase.cardata, ["Acceleration"  "MPG"] );
            testCase.verifyTrue( value )
            
            value = util.isvar( testCase.cardata, ["Acceleration"  "this"] );
            testCase.verifyFalse( value )
            
        end %function
        
        
        function isnormal( testCase )
        
        end %function
        
        function isnumeric( testCase )
        
        end %function
        
        function isconstant( testCase )
        
        end %function
        
            
    end %methods
    
end %classdef 

