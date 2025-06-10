from abc import abstractmethod
from typing import Any, Optional, Type, final


class BaseTransformer:
    """
    Abstract base class for transforming data between different compute frameworks.

    The BaseTransformer defines the interface for converting data between two different
    compute frameworks (e.g., Pandas DataFrame to PyArrow Table). It handles the logic
    for determining the direction of transformation and provides methods for checking
    if the required frameworks are available.

    To implement a new transformer:
    1. Subclass BaseTransformer
    2. Implement all abstract methods
    3. Define the source and target frameworks
    4. Implement the transformation logic in both directions

    The transformer will be automatically discovered and registered with the
    ComputeFrameworkTransformer.
    """

    @final
    @classmethod
    def check_imports(cls) -> bool:
        """
        Check if both frameworks are properly implemented and their modules can be imported.
        """
        # check implementation
        if cls.framework() == NotImplementedError or cls.other_framework() == NotImplementedError:
            return False
        # check if module is installed/imported
        if cls.check_fw_import() is False or cls.check_other_fw_import() is False:
            return False
        return True

    @final
    @classmethod
    def check_fw_import(cls) -> bool:
        try:
            cls.import_fw()
            return True
        except ImportError:
            return False

    @final
    @classmethod
    def check_other_fw_import(cls) -> bool:
        try:
            cls.import_other_fw()
            return True
        except ImportError:
            return False

    @final
    @classmethod
    def identify_orientation(cls, framework: Type[Any], other_framework: Type[Any]) -> str | None:
        """
        Determine the direction of transformation between two frameworks.

        This method identifies whether the transformation should be from the primary
        framework to the secondary framework ("left") or vice versa ("right").
        """
        if framework == other_framework:
            raise ValueError(f"How did you get here? Framework {framework} and {other_framework} are the same")

        cls_framework = cls.framework()
        cls_other_framework = cls.other_framework()

        if framework in (cls_framework, cls_other_framework) and other_framework in (
            cls_framework,
            cls_other_framework,
        ):
            if framework == cls_framework and other_framework == cls_other_framework:
                return "left"
            if framework == cls_other_framework and other_framework == cls_framework:
                return "right"

            raise ValueError(f"Framework {framework} or {other_framework} not supported by {cls.__name__}")

        return None

    @final
    @classmethod
    def transform(
        cls, framework: Type[Any], other_framework: Type[Any], data: Any, framework_connection_object: Optional[Any]
    ) -> Any:
        """Transform data from one framework to another."""

        orientation = cls.identify_orientation(framework, other_framework)

        if orientation is None:
            raise ValueError(
                f"How did you get here? Framework {framework} or {other_framework} not supported by {cls.__name__}"
            )

        if orientation == "left":
            return cls.transform_fw_to_other_fw(data)
        if orientation == "right":
            return cls.transform_other_fw_to_fw(data, framework_connection_object)

        raise ValueError(
            f"How did you get here after right/left? Framework {framework} or {other_framework} not supported by {cls.__name__}"
        )

    @classmethod
    @abstractmethod
    def framework(cls) -> Any:
        """
        Get the primary framework type that this transformer supports.

        Returns:
            Any: The primary framework type (e.g., pd.DataFrame)
        """
        return NotImplementedError

    @classmethod
    @abstractmethod
    def other_framework(cls) -> Any:
        """
        Get the secondary framework type that this transformer supports.

        Returns:
            Any: The secondary framework type (e.g., pa.Table)
        """
        return NotImplementedError

    @classmethod
    @abstractmethod
    def import_fw(cls) -> None:
        """
        Import the primary framework module.

        This method should import the module for the primary framework.
        It will be called to check if the module is available.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def import_other_fw(cls) -> None:
        """
        Import the secondary framework module.

        This method should import the module for the secondary framework.
        It will be called to check if the module is available.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def transform_fw_to_other_fw(cls, data: Any) -> Any:
        """
        Transform data from the primary framework to the secondary framework.

        Args:
            data: Data in the primary framework format

        Returns:
            Any: Transformed data in the secondary framework format
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Optional[Any] = None) -> Any:
        """
        Transform data from the secondary framework to the primary framework.

        Args:
            data: Data in the secondary framework format

        Returns:
            Any: Transformed data in the primary framework format
        """
        raise NotImplementedError
