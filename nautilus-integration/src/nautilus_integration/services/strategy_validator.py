"""
Strategy Validation Pipeline

This module provides comprehensive validation for translated Nautilus strategies,
including compilation checks, safety validation, parameter validation, and
performance estimation.
"""

import ast
import asyncio
import importlib.util
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

from nautilus_integration.core.config import NautilusConfig
from nautilus_integration.core.nautilus_logging import (
    get_correlation_id,
    get_logger,
    log_error_with_context,
    with_correlation_id,
)
from nautilus_integration.services.strategy_translation import (
    F6StrategyDefinition,
    StrategyTranslationResult,
)


class ValidationRule(BaseModel):
    """Validation rule definition."""
    
    rule_id: str
    name: str
    description: str
    severity: str  # 'error', 'warning', 'info'
    category: str  # 'safety', 'performance', 'compliance', 'style'
    enabled: bool = True


class ValidationResult(BaseModel):
    """Result of validation check."""
    
    rule_id: str
    passed: bool
    severity: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    line_number: Optional[int] = None
    column_number: Optional[int] = None


class ComprehensiveValidationReport(BaseModel):
    """Comprehensive validation report."""
    
    validation_id: str = Field(default_factory=lambda: str(uuid4()))
    strategy_id: str
    translation_id: str
    
    # Overall status
    overall_passed: bool = False
    critical_errors: int = 0
    warnings: int = 0
    info_messages: int = 0
    
    # Validation results by category
    safety_results: List[ValidationResult] = Field(default_factory=list)
    performance_results: List[ValidationResult] = Field(default_factory=list)
    compliance_results: List[ValidationResult] = Field(default_factory=list)
    style_results: List[ValidationResult] = Field(default_factory=list)
    
    # Code analysis
    complexity_score: float = 0.0
    maintainability_score: float = 0.0
    performance_score: float = 0.0
    
    # Recommendations
    recommendations: List[str] = Field(default_factory=list)
    
    # Metadata
    validation_time: float = 0.0
    rules_applied: int = 0


class StrategyValidationService:
    """
    Comprehensive strategy validation service.
    
    This service provides multi-layered validation including:
    - Syntax and compilation validation
    - Safety and security checks
    - Performance analysis
    - Code quality assessment
    - Compliance verification
    - Best practices validation
    """
    
    def __init__(self, config: NautilusConfig):
        """
        Initialize strategy validation service.
        
        Args:
            config: NautilusTrader integration configuration
        """
        self.config = config
        self.logger = get_logger("nautilus_integration.strategy_validator")
        
        # Validation rules
        self._validation_rules = self._initialize_validation_rules()
        self._enabled_rules = {rule.rule_id: rule for rule in self._validation_rules if rule.enabled}
        
        # Code analysis tools
        self._ast_analyzer = ASTAnalyzer()
        self._security_analyzer = SecurityAnalyzer()
        self._performance_analyzer = PerformanceAnalyzer()
        
        self.logger.info(
            "Strategy Validation Service initialized",
            rules_count=len(self._validation_rules),
            enabled_rules_count=len(self._enabled_rules),
        )
    
    async def validate_strategy(
        self,
        translation_result: StrategyTranslationResult,
        f6_definition: F6StrategyDefinition
    ) -> ComprehensiveValidationReport:
        """
        Perform comprehensive validation of translated strategy.
        
        Args:
            translation_result: Strategy translation result
            f6_definition: Original F6 strategy definition
            
        Returns:
            Comprehensive validation report
        """
        with with_correlation_id() as correlation_id:
            start_time = asyncio.get_event_loop().time()
            
            self.logger.info(
                "Starting comprehensive strategy validation",
                strategy_id=f6_definition.strategy_id,
                translation_id=translation_result.translation_id,
            )
            
            try:
                # Initialize validation report
                report = ComprehensiveValidationReport(
                    strategy_id=f6_definition.strategy_id,
                    translation_id=translation_result.translation_id,
                )
                
                # Step 1: Syntax and compilation validation
                compilation_results = await self._validate_compilation(
                    translation_result.generated_code
                )
                
                # Step 2: Safety and security validation
                safety_results = await self._validate_safety(
                    translation_result.generated_code,
                    f6_definition
                )
                report.safety_results.extend(safety_results)
                
                # Step 3: Performance validation
                performance_results = await self._validate_performance(
                    translation_result.generated_code,
                    f6_definition
                )
                report.performance_results.extend(performance_results)
                
                # Step 4: Compliance validation
                compliance_results = await self._validate_compliance(
                    translation_result.generated_code,
                    f6_definition
                )
                report.compliance_results.extend(compliance_results)
                
                # Step 5: Code style validation
                style_results = await self._validate_code_style(
                    translation_result.generated_code
                )
                report.style_results.extend(style_results)
                
                # Step 6: Calculate scores and metrics
                await self._calculate_validation_scores(report, translation_result.generated_code)
                
                # Step 7: Generate recommendations
                report.recommendations = await self._generate_recommendations(report, f6_definition)
                
                # Finalize report
                report.validation_time = asyncio.get_event_loop().time() - start_time
                report.rules_applied = len(self._enabled_rules)
                
                # Determine overall status
                report.critical_errors = sum(
                    1 for results in [report.safety_results, report.performance_results, 
                                    report.compliance_results, report.style_results]
                    for result in results
                    if result.severity == 'error' and not result.passed
                )
                
                report.warnings = sum(
                    1 for results in [report.safety_results, report.performance_results,
                                    report.compliance_results, report.style_results]
                    for result in results
                    if result.severity == 'warning' and not result.passed
                )
                
                report.overall_passed = report.critical_errors == 0
                
                self.logger.info(
                    "Strategy validation completed",
                    strategy_id=f6_definition.strategy_id,
                    validation_id=report.validation_id,
                    overall_passed=report.overall_passed,
                    critical_errors=report.critical_errors,
                    warnings=report.warnings,
                    validation_time=report.validation_time,
                )
                
                return report
                
            except Exception as error:
                log_error_with_context(
                    self.logger,
                    error,
                    {
                        "operation": "validate_strategy",
                        "strategy_id": f6_definition.strategy_id,
                        "translation_id": translation_result.translation_id,
                        "correlation_id": correlation_id,
                    },
                    "Failed to validate strategy"
                )
                
                # Return error report
                error_report = ComprehensiveValidationReport(
                    strategy_id=f6_definition.strategy_id,
                    translation_id=translation_result.translation_id,
                    overall_passed=False,
                    critical_errors=1,
                )
                
                error_result = ValidationResult(
                    rule_id="validation_error",
                    passed=False,
                    severity="error",
                    message=f"Validation failed: {error}",
                )
                error_report.safety_results.append(error_result)
                
                return error_report
    
    async def validate_parameter_mapping(
        self,
        f6_definition: F6StrategyDefinition,
        parameter_mapping: Dict[str, str]
    ) -> List[ValidationResult]:
        """
        Validate parameter mapping between F6 and Nautilus.
        
        Args:
            f6_definition: F6 strategy definition
            parameter_mapping: Parameter mapping dictionary
            
        Returns:
            List of validation results
        """
        results = []
        
        try:
            self.logger.debug(
                "Validating parameter mapping",
                strategy_id=f6_definition.strategy_id,
                mapping_count=len(parameter_mapping),
            )
            
            # Check for required parameters
            required_params = self._get_required_parameters_for_family(f6_definition.family)
            for param in required_params:
                if param not in f6_definition.parameters:
                    results.append(ValidationResult(
                        rule_id="missing_required_parameter",
                        passed=False,
                        severity="error",
                        message=f"Required parameter missing: {param}",
                        details={"parameter": param, "family": f6_definition.family}
                    ))
            
            # Validate parameter types and ranges
            for f6_param, f6_value in f6_definition.parameters.items():
                validation_result = self._validate_parameter_value(
                    f6_param, f6_value, f6_definition.family
                )
                if validation_result:
                    results.append(validation_result)
            
            # Check for parameter naming conflicts
            nautilus_params = set(parameter_mapping.values())
            if len(nautilus_params) != len(parameter_mapping):
                results.append(ValidationResult(
                    rule_id="parameter_naming_conflict",
                    passed=False,
                    severity="warning",
                    message="Parameter naming conflicts detected in mapping",
                    details={"mapping": parameter_mapping}
                ))
            
            self.logger.debug(
                "Parameter mapping validation completed",
                strategy_id=f6_definition.strategy_id,
                results_count=len(results),
                errors=sum(1 for r in results if r.severity == 'error' and not r.passed),
            )
            
        except Exception as error:
            self.logger.error(
                "Parameter mapping validation failed",
                strategy_id=f6_definition.strategy_id,
                error=str(error),
            )
            
            results.append(ValidationResult(
                rule_id="parameter_validation_error",
                passed=False,
                severity="error",
                message=f"Parameter validation failed: {error}",
            ))
        
        return results
    
    async def _validate_compilation(self, code: str) -> List[ValidationResult]:
        """Validate code compilation."""
        results = []
        
        try:
            # Syntax check
            try:
                ast.parse(code)
                results.append(ValidationResult(
                    rule_id="syntax_check",
                    passed=True,
                    severity="error",
                    message="Syntax validation passed",
                ))
            except SyntaxError as error:
                results.append(ValidationResult(
                    rule_id="syntax_check",
                    passed=False,
                    severity="error",
                    message=f"Syntax error: {error.msg}",
                    line_number=error.lineno,
                    column_number=error.offset,
                ))
            
            # Compilation check
            try:
                compile(code, '<generated_strategy>', 'exec')
                results.append(ValidationResult(
                    rule_id="compilation_check",
                    passed=True,
                    severity="error",
                    message="Compilation validation passed",
                ))
            except Exception as error:
                results.append(ValidationResult(
                    rule_id="compilation_check",
                    passed=False,
                    severity="error",
                    message=f"Compilation error: {error}",
                ))
            
        except Exception as error:
            results.append(ValidationResult(
                rule_id="compilation_validation_error",
                passed=False,
                severity="error",
                message=f"Compilation validation failed: {error}",
            ))
        
        return results
    
    async def _validate_safety(
        self,
        code: str,
        f6_definition: F6StrategyDefinition
    ) -> List[ValidationResult]:
        """Validate code safety and security."""
        results = []
        
        try:
            # Security analysis
            security_results = await self._security_analyzer.analyze(code)
            results.extend(security_results)
            
            # Check for dangerous imports
            dangerous_imports = [
                'os', 'sys', 'subprocess', 'eval', 'exec', 'compile',
                'importlib', '__import__', 'globals', 'locals'
            ]
            
            for dangerous_import in dangerous_imports:
                if f'import {dangerous_import}' in code or f'from {dangerous_import}' in code:
                    results.append(ValidationResult(
                        rule_id="dangerous_import",
                        passed=False,
                        severity="error",
                        message=f"Dangerous import detected: {dangerous_import}",
                        details={"import": dangerous_import}
                    ))
            
            # Check for file operations
            file_operations = ['open(', 'file(', 'with open']
            for operation in file_operations:
                if operation in code:
                    results.append(ValidationResult(
                        rule_id="file_operation",
                        passed=False,
                        severity="warning",
                        message=f"File operation detected: {operation}",
                        details={"operation": operation}
                    ))
            
            # Check for network operations
            network_patterns = ['urllib', 'requests', 'socket', 'http']
            for pattern in network_patterns:
                if pattern in code:
                    results.append(ValidationResult(
                        rule_id="network_operation",
                        passed=False,
                        severity="warning",
                        message=f"Network operation detected: {pattern}",
                        details={"pattern": pattern}
                    ))
            
            # Validate risk parameters
            risk_validation = self._validate_risk_parameters(f6_definition)
            results.extend(risk_validation)
            
        except Exception as error:
            results.append(ValidationResult(
                rule_id="safety_validation_error",
                passed=False,
                severity="error",
                message=f"Safety validation failed: {error}",
            ))
        
        return results
    
    async def _validate_performance(
        self,
        code: str,
        f6_definition: F6StrategyDefinition
    ) -> List[ValidationResult]:
        """Validate code performance characteristics."""
        results = []
        
        try:
            # Performance analysis
            performance_results = await self._performance_analyzer.analyze(code, f6_definition)
            results.extend(performance_results)
            
            # Check for potential performance issues
            performance_patterns = {
                'nested_loops': ['for.*for', 'while.*while'],
                'inefficient_operations': ['list.*in.*list', '.*\.append.*in.*for'],
                'memory_leaks': ['global ', 'globals()', 'locals()'],
            }
            
            for issue_type, patterns in performance_patterns.items():
                for pattern in patterns:
                    if pattern in code:
                        results.append(ValidationResult(
                            rule_id=f"performance_{issue_type}",
                            passed=False,
                            severity="warning",
                            message=f"Potential performance issue: {issue_type}",
                            details={"pattern": pattern, "issue_type": issue_type}
                        ))
            
        except Exception as error:
            results.append(ValidationResult(
                rule_id="performance_validation_error",
                passed=False,
                severity="error",
                message=f"Performance validation failed: {error}",
            ))
        
        return results
    
    async def _validate_compliance(
        self,
        code: str,
        f6_definition: F6StrategyDefinition
    ) -> List[ValidationResult]:
        """Validate regulatory and compliance requirements."""
        results = []
        
        try:
            # Check for required Nautilus Strategy methods
            required_methods = ['on_start', 'on_stop', 'on_bar']
            for method in required_methods:
                if f'def {method}(' not in code:
                    results.append(ValidationResult(
                        rule_id="missing_required_method",
                        passed=False,
                        severity="error",
                        message=f"Required Nautilus method missing: {method}",
                        details={"method": method}
                    ))
            
            # Check for proper error handling
            if 'try:' in code and 'except Exception as error:' not in code:
                results.append(ValidationResult(
                    rule_id="inadequate_error_handling",
                    passed=False,
                    severity="warning",
                    message="Inadequate error handling detected",
                ))
            
            # Check for logging requirements
            if 'self.log.' not in code:
                results.append(ValidationResult(
                    rule_id="missing_logging",
                    passed=False,
                    severity="warning",
                    message="Strategy should include logging statements",
                ))
            
            # Validate strategy inherits from correct base class
            if 'class' in code and 'Strategy)' not in code:
                results.append(ValidationResult(
                    rule_id="incorrect_base_class",
                    passed=False,
                    severity="error",
                    message="Strategy must inherit from Nautilus Strategy class",
                ))
            
        except Exception as error:
            results.append(ValidationResult(
                rule_id="compliance_validation_error",
                passed=False,
                severity="error",
                message=f"Compliance validation failed: {error}",
            ))
        
        return results
    
    async def _validate_code_style(self, code: str) -> List[ValidationResult]:
        """Validate code style and best practices."""
        results = []
        
        try:
            # AST analysis for code quality
            ast_results = await self._ast_analyzer.analyze(code)
            results.extend(ast_results)
            
            # Check for docstrings
            if '"""' not in code and "'''" not in code:
                results.append(ValidationResult(
                    rule_id="missing_docstrings",
                    passed=False,
                    severity="info",
                    message="Consider adding docstrings for better documentation",
                ))
            
            # Check for magic numbers
            import re
            magic_numbers = re.findall(r'\b\d+\.\d+\b|\b\d+\b', code)
            if len(magic_numbers) > 10:  # Arbitrary threshold
                results.append(ValidationResult(
                    rule_id="too_many_magic_numbers",
                    passed=False,
                    severity="info",
                    message="Consider using named constants instead of magic numbers",
                    details={"count": len(magic_numbers)}
                ))
            
            # Check line length (simplified)
            long_lines = [i for i, line in enumerate(code.split('\n'), 1) if len(line) > 120]
            if long_lines:
                results.append(ValidationResult(
                    rule_id="long_lines",
                    passed=False,
                    severity="info",
                    message=f"Lines exceed 120 characters: {len(long_lines)} lines",
                    details={"line_numbers": long_lines[:5]}  # Show first 5
                ))
            
        except Exception as error:
            results.append(ValidationResult(
                rule_id="style_validation_error",
                passed=False,
                severity="error",
                message=f"Style validation failed: {error}",
            ))
        
        return results
    
    async def _calculate_validation_scores(
        self,
        report: ComprehensiveValidationReport,
        code: str
    ) -> None:
        """Calculate validation scores and metrics."""
        try:
            # Complexity score (simplified cyclomatic complexity)
            complexity = self._calculate_complexity(code)
            report.complexity_score = min(100.0, max(0.0, 100.0 - complexity * 5))
            
            # Maintainability score
            maintainability = self._calculate_maintainability(code, report)
            report.maintainability_score = maintainability
            
            # Performance score
            performance = self._calculate_performance_score(report)
            report.performance_score = performance
            
        except Exception as error:
            self.logger.warning(
                "Failed to calculate validation scores",
                error=str(error),
            )
    
    async def _generate_recommendations(
        self,
        report: ComprehensiveValidationReport,
        f6_definition: F6StrategyDefinition
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        try:
            # Safety recommendations
            safety_errors = [r for r in report.safety_results if not r.passed and r.severity == 'error']
            if safety_errors:
                recommendations.append("Address critical safety issues before deployment")
            
            # Performance recommendations
            if report.performance_score < 70:
                recommendations.append("Consider optimizing strategy performance")
            
            # Complexity recommendations
            if report.complexity_score < 60:
                recommendations.append("Consider simplifying strategy logic for better maintainability")
            
            # Strategy-specific recommendations
            if f6_definition.family == 'trend' and report.performance_score < 80:
                recommendations.append("Trend strategies benefit from efficient indicator calculations")
            
            if f6_definition.complexity == 'high' and report.maintainability_score < 70:
                recommendations.append("High complexity strategies require extensive testing")
            
        except Exception as error:
            self.logger.warning(
                "Failed to generate recommendations",
                error=str(error),
            )
        
        return recommendations
    
    def _initialize_validation_rules(self) -> List[ValidationRule]:
        """Initialize validation rules."""
        return [
            ValidationRule(
                rule_id="syntax_check",
                name="Syntax Validation",
                description="Validate Python syntax correctness",
                severity="error",
                category="safety"
            ),
            ValidationRule(
                rule_id="compilation_check",
                name="Compilation Validation",
                description="Validate code compilation",
                severity="error",
                category="safety"
            ),
            ValidationRule(
                rule_id="dangerous_import",
                name="Dangerous Import Check",
                description="Check for potentially dangerous imports",
                severity="error",
                category="safety"
            ),
            ValidationRule(
                rule_id="missing_required_method",
                name="Required Method Check",
                description="Validate required Nautilus Strategy methods",
                severity="error",
                category="compliance"
            ),
            ValidationRule(
                rule_id="performance_nested_loops",
                name="Nested Loop Check",
                description="Check for potentially inefficient nested loops",
                severity="warning",
                category="performance"
            ),
            ValidationRule(
                rule_id="missing_logging",
                name="Logging Check",
                description="Validate presence of logging statements",
                severity="warning",
                category="compliance"
            ),
        ]
    
    def _get_required_parameters_for_family(self, family: str) -> List[str]:
        """Get required parameters for strategy family."""
        family_requirements = {
            'trend': ['fast_period', 'slow_period'],
            'mean_reversion': ['lookback_period', 'entry_threshold'],
            'momentum': ['lookback_months'],
            'volatility': ['bb_period', 'bb_std'],
        }
        return family_requirements.get(family, [])
    
    def _validate_parameter_value(
        self,
        param_name: str,
        value: Any,
        family: str
    ) -> Optional[ValidationResult]:
        """Validate individual parameter value."""
        # Parameter validation logic
        if param_name.endswith('_period') and isinstance(value, int):
            if value <= 0 or value > 1000:
                return ValidationResult(
                    rule_id="invalid_parameter_range",
                    passed=False,
                    severity="error",
                    message=f"Parameter {param_name} out of valid range: {value}",
                    details={"parameter": param_name, "value": value}
                )
        
        return None
    
    def _validate_risk_parameters(self, definition: F6StrategyDefinition) -> List[ValidationResult]:
        """Validate risk management parameters."""
        results = []
        
        if definition.max_position_size > 0.5:
            results.append(ValidationResult(
                rule_id="high_position_size",
                passed=False,
                severity="warning",
                message=f"High max position size: {definition.max_position_size}",
                details={"value": definition.max_position_size}
            ))
        
        if definition.max_leverage > 5.0:
            results.append(ValidationResult(
                rule_id="high_leverage",
                passed=False,
                severity="warning",
                message=f"High leverage: {definition.max_leverage}",
                details={"value": definition.max_leverage}
            ))
        
        return results
    
    def _calculate_complexity(self, code: str) -> float:
        """Calculate code complexity score."""
        try:
            tree = ast.parse(code)
            complexity = 1  # Base complexity
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.Try)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
            
            return complexity
            
        except Exception:
            return 10.0  # Default high complexity if analysis fails
    
    def _calculate_maintainability(self, code: str, report: ComprehensiveValidationReport) -> float:
        """Calculate maintainability score."""
        base_score = 100.0
        
        # Deduct for errors and warnings
        base_score -= report.critical_errors * 20
        base_score -= report.warnings * 5
        
        # Deduct for complexity
        if report.complexity_score < 70:
            base_score -= 10
        
        return max(0.0, min(100.0, base_score))
    
    def _calculate_performance_score(self, report: ComprehensiveValidationReport) -> float:
        """Calculate performance score."""
        base_score = 100.0
        
        # Deduct for performance issues
        performance_errors = [r for r in report.performance_results if not r.passed]
        base_score -= len(performance_errors) * 10
        
        return max(0.0, min(100.0, base_score))


class ASTAnalyzer:
    """AST-based code analyzer."""
    
    async def analyze(self, code: str) -> List[ValidationResult]:
        """Analyze code using AST."""
        results = []
        
        try:
            tree = ast.parse(code)
            
            # Count function definitions
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            if len(functions) > 20:
                results.append(ValidationResult(
                    rule_id="too_many_functions",
                    passed=False,
                    severity="info",
                    message=f"Many functions defined: {len(functions)}",
                    details={"count": len(functions)}
                ))
            
        except Exception as error:
            results.append(ValidationResult(
                rule_id="ast_analysis_error",
                passed=False,
                severity="warning",
                message=f"AST analysis failed: {error}",
            ))
        
        return results


class SecurityAnalyzer:
    """Security-focused code analyzer."""
    
    async def analyze(self, code: str) -> List[ValidationResult]:
        """Analyze code for security issues."""
        results = []
        
        # Check for potential security issues
        security_patterns = {
            'eval_usage': ['eval(', 'exec('],
            'import_manipulation': ['__import__(', 'importlib'],
            'system_access': ['os.system', 'subprocess'],
        }
        
        for issue_type, patterns in security_patterns.items():
            for pattern in patterns:
                if pattern in code:
                    results.append(ValidationResult(
                        rule_id=f"security_{issue_type}",
                        passed=False,
                        severity="error",
                        message=f"Security issue detected: {issue_type}",
                        details={"pattern": pattern}
                    ))
        
        return results


class PerformanceAnalyzer:
    """Performance-focused code analyzer."""
    
    async def analyze(
        self,
        code: str,
        f6_definition: F6StrategyDefinition
    ) -> List[ValidationResult]:
        """Analyze code for performance issues."""
        results = []
        
        # Strategy-specific performance checks
        if f6_definition.family == 'trend':
            if 'for' in code and 'range(' in code:
                results.append(ValidationResult(
                    rule_id="trend_performance_loop",
                    passed=False,
                    severity="info",
                    message="Consider vectorized operations for trend calculations",
                ))
        
        return results