#pragma once

#include <glad/glad.h>

void initGLDebugOutput();


class GLShader
{
    GLuint m_GLId;
    typedef std::unique_ptr<char[]> CharBuffer;
public:
    GLShader(GLenum type) : m_GLId(glCreateShader(type)) {
    }

    ~GLShader() {
        glDeleteShader(m_GLId);
    }

    GLShader(const GLShader&) = delete;

    GLShader& operator =(const GLShader&) = delete;

    GLShader(GLShader&& rvalue) : m_GLId(rvalue.m_GLId) {
        rvalue.m_GLId = 0;
    }

    GLShader& operator =(GLShader&& rvalue) {
        this->~GLShader();
        m_GLId = rvalue.m_GLId;
        rvalue.m_GLId = 0;
        return *this;
    }

    GLuint glId() const {
        return m_GLId;
    }

    void setSource(const GLchar* src) {
        glShaderSource(m_GLId, 1, &src, 0);
    }

    void setSource(const std::string& src) {
        setSource(src.c_str());
    }

    bool compile() {
        glCompileShader(m_GLId);
        return getCompileStatus();
    }

    bool getCompileStatus() const {
        GLint status;
        glGetShaderiv(m_GLId, GL_COMPILE_STATUS, &status);
        return status == GL_TRUE;
    }

    std::string getInfoLog() const {
        GLint logLength;
        glGetShaderiv(m_GLId, GL_INFO_LOG_LENGTH, &logLength);

        CharBuffer buffer(new char[logLength]);
        glGetShaderInfoLog(m_GLId, logLength, 0, buffer.get());

        return std::string(buffer.get());
    }
};

class GLProgram 
{
    GLuint m_GLId;
    typedef std::unique_ptr<char[]> CharBuffer;
public:
    GLProgram() : m_GLId(glCreateProgram()) {
    }

    ~GLProgram() {
        glDeleteProgram(m_GLId);
    }

    GLProgram(const GLProgram&) = delete;

    GLProgram& operator =(const GLProgram&) = delete;

    GLProgram(GLProgram&& rvalue) : m_GLId(rvalue.m_GLId) {
        rvalue.m_GLId = 0;
    }

    GLProgram& operator =(GLProgram&& rvalue) {
        this->~GLProgram();
        m_GLId = rvalue.m_GLId;
        rvalue.m_GLId = 0;
        return *this;
    }

    GLuint glId() const {
        return m_GLId;
    }

    void attachShader(const GLShader& shader) {
        glAttachShader(m_GLId, shader.glId());
    }

    bool link() {
        glLinkProgram(m_GLId);
        return getLinkStatus();
    }

    bool getLinkStatus() const {
        GLint linkStatus;
        glGetProgramiv(m_GLId, GL_LINK_STATUS, &linkStatus);
        return linkStatus == GL_TRUE;
    }

    std::string getInfoLog() const {
        GLint logLength;
        glGetProgramiv(m_GLId, GL_INFO_LOG_LENGTH, &logLength);

        CharBuffer buffer(new char[logLength]);
        glGetProgramInfoLog(m_GLId, logLength, 0, buffer.get());

        return std::string(buffer.get());
    }

    void use() const {
        glUseProgram(m_GLId);
    }

    GLint getUniformLocation(const GLchar* name) const {
        GLint location = glGetUniformLocation(m_GLId, name);
        return location;
    }

    GLint getAttribLocation(const GLchar* name) const {
        GLint location = glGetAttribLocation(m_GLId, name);
        return location;
    }

    void bindAttribLocation(GLuint index, const GLchar* name) const {
        glBindAttribLocation(m_GLId, index, name);
    }
};

class GLImageRenderer
{
public:
    GLImageRenderer();

    ~GLImageRenderer();

    void drawRGBImage(const float* colors, size_t width, size_t height);

    void drawWeightedRGBImage(const float* colors, size_t width, size_t height, const float weight[3]);

private:
    GLProgram m_Program;
    GLuint m_TriangleVBO;
    GLuint m_TriangleVAO;
    GLuint m_TextureObject;
    size_t m_nTexPixelCount;
    GLuint m_SamplerObject;
    GLint m_uWeight;
};