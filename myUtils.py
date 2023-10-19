"""
This file contains utility functions for the project.

Author:
    Tom Planche
"""
import matplotlib.pyplot as plt
import numpy as np


def angleBetweenVectors(
        vector1: np.array,
        vector2: np.array,
        degrees: bool = True,
        roundTo: int = 2
):
    """
    Calculate the angle in radians between two vectors.

    Args:
        vector1 (numpy.ndarray): The first vector as a numpy array.
        vector2 (numpy.ndarray): The second vector as a numpy array.
        degrees (bool): Whether to return the angle in degrees or radians.
        roundTo (int): The number of decimal places to round the angle to.

    Returns:
        float: The angle in radians between the two vectors.

    Raises:
        ValueError: If either of the input vectors is not a 2D numpy array.
    """
    # Check if the input vectors are 2D numpy arrays
    if vector1.ndim != 2 or vector2.ndim != 2:
        raise ValueError("Input vectors must be 2D numpy arrays.")

    # Calculate the dot product of the two vectors
    dot_product = np.dot(vector1.T, vector2)

    # Calculate the magnitudes (lengths) of the vectors
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Calculate the cosine of the angle between the vectors
    cosine_theta = dot_product / (magnitude1 * magnitude2)

    # Calculate the angle in radians
    theta_radians = np.arccos(cosine_theta)

    if not degrees:
        return theta_radians

    return round(np.rad2deg(theta_radians)[0][0], roundTo)


def calculateLimits(
        vectors: list[np.array] | list,
        origin: list[float] = [0., 0.],
        coefficient: float = 1.1,
        center: bool = True
) -> list[[float]]:
    """
    This function calculates the limits of the plot.

    Args:
        origin: The origin of the plot.
        vectors: The vectors to calculate the limits of.
        coefficient: The coefficient to apply to the limits.
        center: Whether to center the plot or not.

    Returns:
        The limits of the plot.
    """
    [xMin, xMax], [yMin, yMax] = origin, origin

    xMin = min(xMin, min([vector[0][0] for vector in vectors]))
    xMax = max(xMax, max([vector[0][0] for vector in vectors]))
    yMin = min(yMin, min([vector[1][0] for vector in vectors]))
    yMax = max(yMax, max([vector[1][0] for vector in vectors]))

    if center:
        xMin = min(xMin, -xMax / 2)
        xMax = max(xMax, -xMin / 2)
        yMin = min(yMin, -yMax / 2)
        yMax = max(yMax, -yMin / 2)

    return [
        [xMin * coefficient, xMax * coefficient],
        [yMin * coefficient, yMax * coefficient]
    ]


def numpyArrayToLatex(
        array: np.array,
        matrixType: str = 'b',
        wrap: bool = False,
        needLatexSigns: bool = True
) -> str:
    """
    Transforms a numpy array to LaTeX code.

    Args:
        array (np.array): The array to transform.
        matrixType ('b' or 'p'): The apparance of the matrix edges.
        wrap (bool): If wrapping is needed.

    Examples:
        testArray = numpy.array([[0.87, -0.5], [0.5, 0.87]])
        print(numpyArrayToLatex(testArray, True))

        >> \begin{pmatrix} 0.87 & -0.5 \\ 0.5 & 0.87 \end{pmatrix}

    Returns:
        (str) The final LaTeX code.
    """
    tab = '\t' if not wrap else ''
    newLine = '\n' if not wrap else ''
    latexSign = ('$$' if not wrap else '$') if needLatexSigns else ''
    matrixType = matrixType if matrixType in ['p', 'b'] else 'b'

    return (
            f'{latexSign}{newLine}\\begin{{{matrixType}matrix}}{newLine}{tab}'
            + f' \\\\{newLine}{tab}'.join(
                    [" & ".join(str(item) for item in list(row)) for row in array]
                )
            + f'{newLine}\end{{{matrixType}matrix}}{newLine}{latexSign}'
    )


def plot2DColumnVectorText(
        _text: str,
        vector: list[np.array],
        coefficient: float = 1.1
):
    """
    This function plots a 2D vector text.

    Args:
        _text: The text to plot.
        vector: The vector to plot.
        coefficient: The coefficient to apply to the vector.

    Returns:

    """
    textX = vector[0][0] * coefficient
    textY = vector[1][0] * coefficient

    # Add the text label
    text = plt.text(
        textX,
        textY,
        _text,
    )

    # Get the bounding box of the text
    bbox = text.get_window_extent()

    # Retrieve the height and width
    height, width = bbox.height, bbox.width

    textX = vector[0] * coefficient
    textY = vector[1] * coefficient

    """
    transform the text bounding box coordinate system from an absolute position
    to a relative position with respect to the plot area.
    """
    # Get the inverse of the current plot transform
    inv = plt.gca().transData.inverted()

    # Transform the four points of the bounding box
    invTrans = inv.transform([[0, 0], [width, height]])

    # Retrieve the new width and height
    invWidth = invTrans[1][0] - invTrans[0][0]
    invHeight = invTrans[1][1] - invTrans[0][1]

    # Determine the new x and y coordinates
    x = textX - invWidth / 2
    y = textY - invHeight / 2

    # Set the text object's new coordinates
    text.set_position([x, y])


