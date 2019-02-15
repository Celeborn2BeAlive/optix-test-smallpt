#include <unordered_map>
#include <vector>
#include <array>
#include <tuple>
#include <iostream>
#include <memory>
#include <string>

#include "glutils.h"

static const std::unordered_map<GLenum, const char *> sourceEnumToString =
{
    { GL_DEBUG_SOURCE_API , "API" },
    { GL_DEBUG_SOURCE_WINDOW_SYSTEM , "WINDOW_SYSTEM" },
    { GL_DEBUG_SOURCE_SHADER_COMPILER , "SHADER_COMPILER" },
    { GL_DEBUG_SOURCE_THIRD_PARTY , "THIRD_PARTY" },
    { GL_DEBUG_SOURCE_APPLICATION , "APPLICATION" },
    { GL_DEBUG_SOURCE_OTHER , "OTHER" }
};

static const std::unordered_map<GLenum, const char *> typeEnumToString =
{
    { GL_DEBUG_TYPE_ERROR , "ERROR" },
    { GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR , "DEPRECATED_BEHAVIOR" },
    { GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR , "UNDEFINED_BEHAVIOR" },
    { GL_DEBUG_TYPE_PORTABILITY , "PORTABILITY" },
    { GL_DEBUG_TYPE_PERFORMANCE , "PERFORMANCE" },
    { GL_DEBUG_TYPE_OTHER , "OTHER" }
};

static const std::unordered_map<GLenum, const char *> severityEnumToString =
{
    { GL_DEBUG_SEVERITY_HIGH , "HIGH" },
    { GL_DEBUG_SEVERITY_MEDIUM , "MEDIUM" },
    { GL_DEBUG_SEVERITY_LOW , "LOW" },
    { GL_DEBUG_SEVERITY_NOTIFICATION , "NOTIFICATION" }
};

// List of message type to ignore for GL Debug Output
static const std::vector<std::tuple<GLenum, GLenum, GLenum>> ignoreList =
{
    std::make_tuple(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION) // Ignore all notifications
};

static std::array<std::tuple<const char *, bool, GLenum>, 6> sourceSelector =
{
    std::make_tuple("API", true, GL_DEBUG_SOURCE_API),
    std::make_tuple("WINDOW_SYSTEM", true, GL_DEBUG_SOURCE_WINDOW_SYSTEM),
    std::make_tuple("SHADER_COMPILER", true, GL_DEBUG_SOURCE_SHADER_COMPILER),
    std::make_tuple("THIRD_PARTY", true, GL_DEBUG_SOURCE_THIRD_PARTY),
    std::make_tuple("APPLICATION", true, GL_DEBUG_SOURCE_APPLICATION),
    std::make_tuple("OTHER", true, GL_DEBUG_SOURCE_OTHER),
};

static std::array<std::tuple<const char *, bool, GLenum>, 6> typeSelector =
{
    std::make_tuple("ERROR", true, GL_DEBUG_TYPE_ERROR),
    std::make_tuple("DEPRECATED_BEHAVIOR", true, GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR),
    std::make_tuple("UNDEFINED_BEHAVIOR", true, GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR),
    std::make_tuple("PORTABILITY", true, GL_DEBUG_TYPE_PORTABILITY),
    std::make_tuple("PERFORMANCE", true, GL_DEBUG_TYPE_PERFORMANCE),
    std::make_tuple("OTHER", true, GL_DEBUG_TYPE_OTHER),
};

static std::array<std::tuple<const char *, bool, GLenum>, 4> severitySelector =
{
    std::make_tuple("HIGH", true, GL_DEBUG_SEVERITY_HIGH),
    std::make_tuple("MEDIUM", true, GL_DEBUG_SEVERITY_MEDIUM),
    std::make_tuple("LOW", true, GL_DEBUG_SEVERITY_LOW),
    std::make_tuple("NOTIFICATION", false, GL_DEBUG_SEVERITY_NOTIFICATION)
};

void logGLDebugInfo(GLenum source, GLenum type, GLuint id, GLenum severity,
    GLsizei length, const GLchar* message, GLvoid* userParam);

void initGLDebugOutput()
{
    glDebugMessageCallback((GLDEBUGPROCARB)logGLDebugInfo, nullptr);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);

    for (const auto & tuple : ignoreList) {
        glDebugMessageControl(std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple), 0, nullptr, GL_FALSE);
    }
}

void logGLDebugInfo(GLenum source, GLenum type, GLuint id, GLenum severity,
    GLsizei length, const GLchar* message, GLvoid* userParam)
{
    const auto findStr = [](GLenum value, const auto & map)
    {
        const auto it = map.find(value);
        if (it == end(map)) {
            return "UNDEFINED";
        }
        return (*it).second;
    };

    const auto sourceStr = findStr(source, sourceEnumToString);
    const auto typeStr = findStr(type, typeEnumToString);
    const auto severityStr = findStr(severity, severityEnumToString);

    std::clog << "OpenGL: " << message << " [source=" << sourceStr << " type=" << typeStr << " severity=" << severityStr << " id=" << id << "]\n\n";
}

template<typename StringType>
GLShader compileShader(GLenum type, StringType&& src) {
    GLShader shader(type);
    shader.setSource(std::forward<StringType>(src));
    if (!shader.compile()) {
        std::cerr << shader.getInfoLog() << std::endl;
        throw std::runtime_error(shader.getInfoLog());
    }
    return shader;
}

inline GLProgram buildProgram(std::initializer_list<GLShader> shaders) {
    GLProgram program;
    for (const auto& shader : shaders) {
        program.attachShader(shader);
    }
    if (!program.link()) {
        std::cerr << program.getInfoLog() << std::endl;
        throw std::runtime_error(program.getInfoLog());
    }

    return program;
}

template<typename VSrc, typename FSrc>
GLProgram buildProgram(VSrc&& vsrc, FSrc&& fsrc) {
    GLShader vs = compileShader(GL_VERTEX_SHADER, std::forward<VSrc>(vsrc));
    GLShader fs = compileShader(GL_FRAGMENT_SHADER, std::forward<FSrc>(fsrc));

    return buildProgram({ std::move(vs), std::move(fs) });
}

template<typename VSrc, typename GSrc, typename FSrc>
GLProgram buildProgram(VSrc&& vsrc, GSrc&& gsrc, FSrc&& fsrc) {
    GLShader vs = compileShader(GL_VERTEX_SHADER, std::forward<VSrc>(vsrc));
    GLShader gs = compileShader(GL_GEOMETRY_SHADER, std::forward<GSrc>(gsrc));
    GLShader fs = compileShader(GL_FRAGMENT_SHADER, std::forward<FSrc>(fsrc));

    return buildProgram({ std::move(vs), std::move(gs), std::move(fs) });
}

template<typename CSrc>
GLProgram buildComputeProgram(CSrc&& src) {
    GLShader cs = compileShader(GL_COMPUTE_SHADER, std::forward<CSrc>(src));
    return buildProgram({ std::move(cs) });;
}

static const char * ImageVertexShader = R"GLSL"(
#version 330

layout(location = 0) in vec2 aPosition;

out vec2 vTexCoords;

void main()
{
    vTexCoords = 0.5 * (aPosition + vec2(1));
    gl_Position = vec4(aPosition, 0, 1);
}

)GLSL"";

static const char * ImageFragmentShader = R"GLSL"(
#version 330

uniform sampler2D uTexture;
uniform vec3 uWeight;

in vec2 vTexCoords;

out vec4 fFragColor;

void main()
{
    fFragColor = vec4(uWeight, 1) * texture(uTexture, vTexCoords);
}

)GLSL"";

GLImageRenderer::GLImageRenderer()
{
    glCreateBuffers(1, &m_TriangleVBO);

    GLfloat data[] = { -1, -1, 3, -1, -1, 3 };
    glNamedBufferStorage(m_TriangleVBO, sizeof(data), data, 0);

    glCreateVertexArrays(1, &m_TriangleVAO);

    glEnableVertexArrayAttrib(m_TriangleVAO, 0);

    glVertexArrayAttribBinding(m_TriangleVAO, 0, 0); // attrib 0 goes to binding 0
    glVertexArrayVertexBuffer(m_TriangleVAO, 0, m_TriangleVBO, 0, 2 * sizeof(float));

    glVertexArrayAttribFormat(m_TriangleVAO, 0, 2, GL_FLOAT, GL_FALSE, 0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    m_Program = buildProgram(ImageVertexShader, ImageFragmentShader);

    const auto uTexture = m_Program.getUniformLocation("uTexture");
    glProgramUniform1i(m_Program.glId(), uTexture, 0);

    m_TextureObject = 0;
    m_nTexPixelCount = 0;

    glCreateSamplers(1, &m_SamplerObject);
    glSamplerParameteri(m_SamplerObject, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glSamplerParameteri(m_SamplerObject, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    m_uWeight = m_Program.getUniformLocation("uWeight");
}

GLImageRenderer::~GLImageRenderer()
{
    glDeleteBuffers(1, &m_TriangleVBO);
    glDeleteVertexArrays(1, &m_TriangleVAO);
}

void GLImageRenderer::drawRGBImage(const float* colors, size_t width, size_t height)
{
    float weight[3] = { 1, 1, 1 };
    drawWeightedRGBImage(colors, width, height, weight);
}

void GLImageRenderer::drawWeightedRGBImage(const float* colors, size_t width, size_t height, const float weight[3])
{
    m_Program.use();

    const auto pixelCount = width * height;
    if (m_nTexPixelCount != pixelCount)
    {
        if (m_TextureObject)
            glDeleteTextures(1, &m_TextureObject);

        glCreateTextures(GL_TEXTURE_2D, 1, &m_TextureObject);
        glTextureStorage2D(m_TextureObject, 1, GL_RGBA32F, width, height);

        m_nTexPixelCount = pixelCount;
    }

    glTextureSubImage2D(m_TextureObject, 0, 0, 0, width, height, GL_RGB, GL_FLOAT, colors);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_TextureObject);
    glBindSampler(0, m_SamplerObject);

    glUniform3fv(m_uWeight, 1, weight);

    glBindVertexArray(m_TriangleVAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}